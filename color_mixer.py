import pandas as pd
import numpy as np
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from math import gcd
from itertools import combinations
import sys
import os
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Попытка импортировать ускоренную Delta E
USE_NUMBA = False
USE_CYTHON = False

try:
    from numba import jit
    USE_NUMBA = True
except ImportError:
    jit = lambda x: x

try:
    from deltae_cython import delta_e_cie2000 as cython_delta_e
    USE_CYTHON = True
except ImportError:
    cython_delta_e = None

# ============================================================
# Быстрая Delta E 2000 (Numba)
# ============================================================
@jit(nopython=True)
def delta_e_cie2000_fast(L1, a1, b1, L2, a2, b2):
    kL, kC, kH = 1.0, 1.0, 1.0
    delta_Lp = L2 - L1
    L_bar = (L1 + L2) / 2.0
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0
    a1p = a1 + (a1 / 2.0) * (1.0 - np.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    a2p = a2 + (a2 / 2.0) * (1.0 - np.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    delta_Cp = C2p - C1p
    C_barp = (C1p + C2p) / 2.0
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    delta_hp = h2p - h1p
    if C1p * C2p != 0:
        if delta_hp > 180.0:
            delta_hp -= 360.0
        elif delta_hp < -180.0:
            delta_hp += 360.0
    H_barp = (h1p + h2p) / 2.0
    if abs(delta_hp) > 180.0:
        H_barp += 180.0
    T = 1.0 - 0.17 * np.cos(np.radians(H_barp - 30.0)) \
        + 0.24 * np.cos(np.radians(2 * H_barp)) \
        + 0.32 * np.cos(np.radians(3 * H_barp + 6.0)) \
        - 0.20 * np.cos(np.radians(4 * H_barp - 63.0))
    delta_Hp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(delta_hp) / 2.0)
    S_L = 1.0 + (0.015 * (L_bar - 50.0)**2) / np.sqrt(20.0 + (L_bar - 50.0)**2)
    S_C = 1.0 + 0.045 * C_barp
    S_H = 1.0 + 0.015 * C_barp * T
    delta_theta = 30.0 * np.exp(-((H_barp - 275.0) / 25.0)**2)
    R_C = 2.0 * np.sqrt(C_barp**7 / (C_barp**7 + 25**7))
    R_T = -R_C * np.sin(2.0 * np.radians(delta_theta))
    delta_E = np.sqrt(
        (delta_Lp / (kL * S_L))**2 +
        (delta_Cp / (kC * S_C))**2 +
        (delta_Hp / (kH * S_H))**2 +
        R_T * (delta_Cp / (kC * S_C)) * (delta_Hp / (kH * S_H))
    )
    return delta_E

def delta_e_cie2000_dispatch(lab1, lab2):
    if USE_CYTHON and cython_delta_e is not None:
        return cython_delta_e(lab1.lab_l, lab1.lab_a, lab1.lab_b,
                              lab2.lab_l, lab2.lab_a, lab2.lab_b)
    elif USE_NUMBA:
        return delta_e_cie2000_fast(lab1.lab_l, lab1.lab_a, lab1.lab_b,
                                    lab2.lab_l, lab2.lab_a, lab2.lab_b)
    else:
        from colormath.color_diff import delta_e_cie2000 as orig
        return orig(lab1, lab2)

# ------------------------------------------------------------
# Загрузка и подготовка данных (без изменений)
# ------------------------------------------------------------
CSV_FILENAME = "paints.csv"
EPSILON = 0.5

def detect_encoding(file_path):
    encodings = ['utf-8-sig', 'utf-8', 'cp1251', 'windows-1251', 'koi8-r', 'latin1']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                f.readline()
            return enc
        except UnicodeDecodeError:
            continue
    return 'latin1'

def detect_delimiter(file_path, encoding):
    with open(file_path, 'r', encoding=encoding) as f:
        first = f.readline()
        return ';' if ';' in first and ',' not in first else ','

def load_paints():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, CSV_FILENAME)
    if not os.path.exists(file_path):
        print(f"❌ Файл {CSV_FILENAME} не найден в папке:\n   {script_dir}")
        sys.exit(1)
    encoding = detect_encoding(file_path)
    delim = detect_delimiter(file_path, encoding)
    df = pd.read_csv(file_path, delimiter=delim, encoding=encoding)
    df.columns = df.columns.str.strip().str.lower()
    required = ['article', 'name', 'hex', 'brand']
    missing = [c for c in required if c not in df.columns]
    if missing:
        other = ';' if delim == ',' else ','
        df = pd.read_csv(file_path, delimiter=other, encoding=encoding)
        df.columns = df.columns.str.strip().str.lower()
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"❌ Ошибка: в CSV нет колонок {required}\n   Найдено: {list(df.columns)}")
            sys.exit(1)
    df['hex'] = df['hex'].astype(str).str.replace('#', '').str.upper()
    df = df.dropna(subset=['hex'])
    print(f"✅ Загружено красок: {len(df)}")
    return df

def hex_to_lab(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return convert_color(sRGBColor(r, g, b), LabColor)

def lab_to_hex(lab):
    rgb = convert_color(lab, sRGBColor)
    r = max(0, min(1, rgb.rgb_r))
    g = max(0, min(1, rgb.rgb_g))
    b = max(0, min(1, rgb.rgb_b))
    return f"{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"

def generate_weights_for_2():
    weights_set = set()
    for denom in range(2, 9):
        for a in range(1, denom):
            b = denom - a
            w1 = a / denom
            w2 = b / denom
            weights_set.add((round(w1, 3), round(w2, 3)))
    return list(weights_set)

def generate_weights_for_3():
    return [(1/3, 1/3, 1/3)]

def mix_colors(lab_list, weights):
    L = sum(w * c.lab_l for w, c in zip(weights, lab_list))
    a = sum(w * c.lab_a for w, c in zip(weights, lab_list))
    b = sum(w * c.lab_b for w, c in zip(weights, lab_list))
    return LabColor(L, a, b)

# ------------------------------------------------------------
# Многопроцессорные функции для перебора
# ------------------------------------------------------------
def process_pair(args, target_lab, paints_df, weights_list):
    idx1, idx2 = args
    paint1 = paints_df.loc[idx1]
    paint2 = paints_df.loc[idx2]
    lab1 = hex_to_lab(paint1['hex'])
    lab2 = hex_to_lab(paint2['hex'])
    results = []
    for w1, w2 in weights_list:
        mixed = mix_colors([lab1, lab2], [w1, w2])
        delta = delta_e_cie2000_dispatch(target_lab, mixed)
        results.append({
            'delta_e': delta,
            'paints': [paint1, paint2],
            'weights': (w1, w2),
            'mixed_lab': mixed
        })
    return results

def process_triplet(args, target_lab, paints_df, weights):
    idx1, idx2, idx3 = args
    paint1 = paints_df.loc[idx1]
    paint2 = paints_df.loc[idx2]
    paint3 = paints_df.loc[idx3]
    lab1 = hex_to_lab(paint1['hex'])
    lab2 = hex_to_lab(paint2['hex'])
    lab3 = hex_to_lab(paint3['hex'])
    mixed = mix_colors([lab1, lab2, lab3], weights)
    delta = delta_e_cie2000_dispatch(target_lab, mixed)
    return {
        'delta_e': delta,
        'paints': [paint1, paint2, paint3],
        'weights': weights,
        'mixed_lab': mixed
    }

def find_best_mix_2(target_lab, paints_df):
    weights_list = generate_weights_for_2()
    pairs = list(combinations(paints_df.index, 2))
    if not pairs:
        return []
    num_cores = max(1, mp.cpu_count() - 1)
    print(f"🖥️ Используем {num_cores} процессоров для перебора 2 красок")
    worker = partial(process_pair, target_lab=target_lab, paints_df=paints_df, weights_list=weights_list)
    all_results = []
    with mp.Pool(processes=num_cores) as pool:
        # Используем imap_unordered с прогресс-баром
        for chunk in tqdm(pool.imap_unordered(worker, pairs), total=len(pairs), desc="Перебор пар"):
            all_results.extend(chunk)
    all_results.sort(key=lambda x: x['delta_e'])
    return all_results[:3]

def find_best_mix_3(target_lab, paints_df):
    weights = (1/3, 1/3, 1/3)
    triplets = list(combinations(paints_df.index, 3))
    if not triplets:
        return []
    num_cores = max(1, mp.cpu_count() - 1)
    print(f"🖥️ Используем {num_cores} процессоров для перебора 3 красок")
    worker = partial(process_triplet, target_lab=target_lab, paints_df=paints_df, weights=weights)
    all_results = []
    with mp.Pool(processes=num_cores) as pool:
        for res in tqdm(pool.imap_unordered(worker, triplets), total=len(triplets), desc="Перебор троек"):
            all_results.append(res)
    all_results.sort(key=lambda x: x['delta_e'])
    return all_results[:3]

# ------------------------------------------------------------
# Остальные функции (точное совпадение, вывод и т.д.)
# ------------------------------------------------------------
def find_exact_matches(target_lab, paints_df):
    matches = []
    for idx, row in paints_df.iterrows():
        lab = hex_to_lab(row['hex'])
        delta = delta_e_cie2000_dispatch(target_lab, lab)
        if delta <= EPSILON:
            matches.append({'paint': row, 'delta_e': delta})
    return matches

def delta_to_percent(delta):
    return max(0.0, min(100.0, 100.0 - delta * 10.0))

def auto_find_mixes(target_lab, paints_df):
    exact = find_exact_matches(target_lab, paints_df)
    if exact:
        print(f"\n✅ Найдено точных совпадений: {len(exact)}")
        return {'type': 'exact', 'result': exact}
    print("\n🎨 Точной краски нет. Пробуем смешать 2 краски...")
    best_2 = find_best_mix_2(target_lab, paints_df)
    if best_2:
        best_delta_2 = best_2[0]['delta_e']
        print(f"Лучшая смесь из 2 красок даёт Delta E={best_delta_2:.2f}")
        if best_delta_2 <= 2.0:
            return {'type': 'mix2', 'result': best_2}
        else:
            print("Точность невысока, пробуем 3 краски...")
    else:
        print("Не найдено смесей из 2 красок, пробуем 3 краски...")
    best_3 = find_best_mix_3(target_lab, paints_df)
    if best_3:
        return {'type': 'mix3', 'result': best_3}
    else:
        return {'type': 'none', 'result': None}

def weights_to_parts(weights):
    from fractions import Fraction
    fracs = [Fraction(w).limit_denominator(8) for w in weights]
    denoms = [f.denominator for f in fracs]
    common_denom = 1
    for d in denoms:
        common_denom = common_denom * d // gcd(common_denom, d)
    parts = [f.numerator * (common_denom // f.denominator) for f in fracs]
    g = parts[0]
    for p in parts[1:]:
        g = gcd(g, p)
    parts = [p // g for p in parts]
    return ':'.join(str(p) for p in parts)

def build_hex_alternatives(df):
    alt = {}
    for _, row in df.iterrows():
        h = row['hex']
        alt.setdefault(h, []).append({'article': row['article'], 'brand': row['brand']})
    return {h: lst for h, lst in alt.items() if len(lst) > 1}

def format_alternatives(hex_code, alt_dict):
    if hex_code in alt_dict:
        alts = alt_dict[hex_code]
        parts = [f"арт.{a['article']}, {a['brand']}" for a in alts]
        return " (" + "; ".join(parts) + ")"
    return ""

def format_paint_with_alternatives(paint_row, alt_dict):
    hex_code = paint_row['hex']
    base = f"{paint_row['brand']} / {paint_row['name']} (арт.{paint_row['article']}) — #{hex_code}"
    return base + format_alternatives(hex_code, alt_dict)

def print_results(target_hex, result_dict, alt_dict):
    print("\n" + "="*70)
    print(f"Целевой цвет: #{target_hex.upper()}")
    print("="*70)
    if result_dict['type'] == 'exact':
        matches = result_dict['result']
        print(f"\n✅ Точное совпадение (Delta E ≤ {EPSILON}):")
        for i, match in enumerate(matches):
            paint = match['paint']
            line = format_paint_with_alternatives(paint, alt_dict)
            if i == 0:
                print(f"   {line}")
            else:
                print(f"   Альтернатива: {line}")
        return
    if result_dict['type'] == 'none':
        print("Не найдено подходящих смесей.")
        return
    top = result_dict['result']
    if not top:
        print("Не найдено подходящих смесей.")
        return
    n = 2 if result_dict['type'] == 'mix2' else 3
    print(f"\n🎨 Лучшие смеси из {n} красок:")
    for i, mix in enumerate(top):
        delta = mix['delta_e']
        real_perc = delta_to_percent(delta)
        parts = weights_to_parts(mix['weights'])
        perc_str = ' + '.join(f"{p*100:.1f}%" for p in mix['weights'])
        print(f"\n--- Вариант {i+1} (совпадение ~{real_perc:.1f}%, Delta E={delta:.2f}) ---")
        print(f"Пропорции (части): {parts}")
        print(f"Пропорции (%): {perc_str}")
        for j, p in enumerate(mix['paints']):
            print(f"  {j+1}. {format_paint_with_alternatives(p, alt_dict)}")
    print("\n* 100% = Delta E=0 (идеал), 0% = Delta E≥10")

# ------------------------------------------------------------
# Основной диалог
# ------------------------------------------------------------
def interactive():
    print("🎨 ПОДБОР СМЕСИ КРАСОК (многопроцессорный, с Numba/Cython)")
    print("="*70)
    if USE_NUMBA:
        print("⚡ Ускорение: Numba JIT включено")
    if USE_CYTHON:
        print("⚡ Ускорение: Cython модуль загружен")
    if not USE_NUMBA and not USE_CYTHON:
        print("⚠️ Ускорение не активно. Установите numba или скомпилируйте Cython модуль.")
    paints = load_paints()
    alt_dict = build_hex_alternatives(paints)
    if alt_dict:
        print(f"📌 Обнаружены дубликаты по HEX: {len(alt_dict)} цветов имеют альтернативы.")
    target = input("\nЦелевой HEX (например FFD700): ").strip().lstrip('#')
    if len(target)!=6 or not all(c in '0123456789ABCDEFabcdef' for c in target):
        print("Неверный HEX.")
        sys.exit(1)
    target = target.upper()
    target_lab = hex_to_lab(target)
    result = auto_find_mixes(target_lab, paints)
    print_results(target, result, alt_dict)

if __name__ == '__main__':
    # Необходимо для multiprocessing на Windows
    mp.freeze_support()
    interactive()
