#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import sqlite3
import numpy as np
import pandas as pd
from math import gcd
from fractions import Fraction
from itertools import combinations
from tqdm import tqdm
import multiprocessing as mp
import traceback
import mixbox

# ---------- Отключаем излишнее логирование (ускорение) ----------
import logging
logging.disable(logging.CRITICAL)

# ---------- Быстрые преобразования RGB <-> LAB (без colormath) ----------
SRGB_TO_XYZ_MATRIX = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])
XYZ_WHITE = np.array([0.95047, 1.00000, 0.95047])

def rgb_to_lab_fast(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    # sRGB -> линейное RGB
    r = r**2.4 if r > 0.04045 else r / 12.92
    g = g**2.4 if g > 0.04045 else g / 12.92
    b = b**2.4 if b > 0.04045 else b / 12.92
    # линейное RGB -> XYZ
    xyz = np.dot(SRGB_TO_XYZ_MATRIX, [r, g, b])
    xyz_n = xyz / XYZ_WHITE
    def f(t):
        return t**(1/3) if t > 0.008856 else (7.787 * t + 16/116)
    L = 116.0 * f(xyz_n[1]) - 16.0
    a = 500.0 * (f(xyz_n[0]) - f(xyz_n[1]))
    b_ = 200.0 * (f(xyz_n[1]) - f(xyz_n[2]))
    return (L, a, b_)

def lab_to_rgb_fast(lab):
    L, a, b = lab
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    def f_inv(t):
        t3 = t**3
        return t3 if t3 > 0.008856 else (t - 16/116) / 7.787
    x = f_inv(fx) * XYZ_WHITE[0]
    y = f_inv(fy) * XYZ_WHITE[1]
    z = f_inv(fz) * XYZ_WHITE[2]
    inv_matrix = np.linalg.inv(SRGB_TO_XYZ_MATRIX)
    rgb_lin = np.dot(inv_matrix, [x, y, z])
    def gamma(c):
        c = max(0.0, min(1.0, c))
        return 12.92 * c if c <= 0.0031308 else 1.055 * (c**(1/2.4)) - 0.055
    r = gamma(rgb_lin[0])
    g = gamma(rgb_lin[1])
    b_ = gamma(rgb_lin[2])
    return (int(round(r * 255)), int(round(g * 255)), int(round(b_ * 255)))

# ---------- Delta E 2000 (Numba, если доступна) ----------
USE_NUMBA = False
try:
    from numba import jit
    USE_NUMBA = True
except ImportError:
    def jit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

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

# ---------- Загрузка базы ----------
DB_FILENAME = "paints.db"
EPSILON = 0.5

def load_paints():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, DB_FILENAME)
    if not os.path.exists(db_path):
        print(f"[X] Файл базы данных {DB_FILENAME} не найден в папке:\n   {script_dir}")
        print("Убедитесь, что вы сконвертировали CSV в SQLite с помощью convert_csv_to_sqlite.py")
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT article, name, hex, brand FROM paints", conn)
    conn.close()
    df['hex'] = df['hex'].astype(str).str.replace('#', '').str.upper()
    df = df.dropna(subset=['hex'])
    print(f"[OK] Загружено красок из SQLite: {len(df)}")
    return df

# ---------- Генерация весов ----------
def generate_weights_for_2(max_denom=6):
    weights_set = set()
    for denom in range(2, max_denom + 1):
        for a in range(1, denom):
            b = denom - a
            w1 = a / denom
            w2 = b / denom
            weights_set.add((round(w1, 3), round(w2, 3)))
    return list(weights_set)

def generate_weights_for_3():
    return [(1/3, 1/3, 1/3)]

# ---------- Предвычисление данных красок ----------
def precompute_paints_data(paints_df):
    paints_data = []
    for idx, row in paints_df.iterrows():
        hex_color = row['hex']
        rgb = (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))
        lab = rgb_to_lab_fast(rgb)
        latent = mixbox.rgb_to_latent(rgb)
        paints_data.append({
            'idx': idx,
            'L': lab[0],
            'a': lab[1],
            'b': lab[2],
            'rgb': rgb,
            'brand': row['brand'],
            'name': row['name'],
            'article': row['article'],
            'hex': hex_color,
            'latent': latent
        })
    return paints_data

# ---------- Обработка пары (оптимизировано) ----------
def process_pair(args, target_L, target_a, target_b, paints_data, weights_list):
    idx1, idx2 = args
    p1 = paints_data[idx1]
    p2 = paints_data[idx2]
    z1 = p1['latent']
    z2 = p2['latent']
    results = []
    for w1, w2 in weights_list:
        z_mix = [0.0] * mixbox.LATENT_SIZE
        for i in range(mixbox.LATENT_SIZE):
            z_mix[i] = w1 * z1[i] + w2 * z2[i]
        rgb_mix = mixbox.latent_to_rgb(z_mix)
        L_mix, a_mix, b_mix = rgb_to_lab_fast(rgb_mix)
        delta = delta_e_cie2000_fast(target_L, target_a, target_b, L_mix, a_mix, b_mix)
        results.append({
            'delta_e': delta,
            'paints': [p1, p2],
            'weights': (w1, w2),
            'mixed_lab': (L_mix, a_mix, b_mix)
        })
    return results

# ---------- Обработка тройки ----------
def process_triplet(args, target_L, target_a, target_b, paints_data, weights):
    idx1, idx2, idx3 = args
    p1 = paints_data[idx1]
    p2 = paints_data[idx2]
    p3 = paints_data[idx3]
    z1 = p1['latent']
    z2 = p2['latent']
    z3 = p3['latent']
    z_mix = [0.0] * mixbox.LATENT_SIZE
    w1, w2, w3 = weights
    for i in range(mixbox.LATENT_SIZE):
        z_mix[i] = w1 * z1[i] + w2 * z2[i] + w3 * z3[i]
    rgb_mix = mixbox.latent_to_rgb(z_mix)
    L_mix, a_mix, b_mix = rgb_to_lab_fast(rgb_mix)
    delta = delta_e_cie2000_fast(target_L, target_a, target_b, L_mix, a_mix, b_mix)
    return {
        'delta_e': delta,
        'paints': [p1, p2, p3],
        'weights': weights,
        'mixed_lab': (L_mix, a_mix, b_mix)
    }

# ---------- Предотбор красок ----------
def select_top_paints(target_lab, paints_data, top_k=200):
    L, a, b = target_lab
    distances = []
    for i, p in enumerate(paints_data):
        d = (p['L'] - L)**2 + (p['a'] - a)**2 + (p['b'] - b)**2
        distances.append((d, i))
    distances.sort(key=lambda x: x[0])
    return [idx for _, idx in distances[:top_k]]

# ---------- Поиск лучших смесей ----------
def find_best_mix_2(target_lab, paints_df, top_k=200):
    target_L, target_a, target_b = target_lab
    weights_list = generate_weights_for_2(max_denom=6)
    full_paints = precompute_paints_data(paints_df)
    selected_indices = select_top_paints((target_L, target_a, target_b), full_paints, top_k)
    paints_data = [full_paints[i] for i in selected_indices]
    n = len(paints_data)
    total_pairs = n * (n - 1) // 2
    print(f"[CPU] Перебор 2 красок (предотбор {top_k} из {len(full_paints)}, всего {total_pairs} пар) с Mixbox")
    top_results = []
    for idx1, idx2 in tqdm(combinations(range(n), 2), total=total_pairs, desc="Перебор пар"):
        results = process_pair((idx1, idx2), target_L, target_a, target_b, paints_data, weights_list)
        for res in results:
            if len(top_results) < 3:
                top_results.append(res)
                top_results.sort(key=lambda x: x['delta_e'])
            else:
                if res['delta_e'] < top_results[-1]['delta_e']:
                    top_results[-1] = res
                    top_results.sort(key=lambda x: x['delta_e'])
    return top_results

def find_best_mix_3(target_lab, paints_df, top_k=100):
    target_L, target_a, target_b = target_lab
    weights = (1/3, 1/3, 1/3)
    full_paints = precompute_paints_data(paints_df)
    selected_indices = select_top_paints((target_L, target_a, target_b), full_paints, top_k)
    paints_data = [full_paints[i] for i in selected_indices]
    n = len(paints_data)
    total_triplets = n * (n - 1) * (n - 2) // 6
    print(f"[CPU] Перебор 3 красок (предотбор {top_k} из {len(full_paints)}, всего {total_triplets} троек) с Mixbox")
    top_results = []
    for idx1, idx2, idx3 in tqdm(combinations(range(n), 3), total=total_triplets, desc="Перебор троек"):
        res = process_triplet((idx1, idx2, idx3), target_L, target_a, target_b, paints_data, weights)
        if len(top_results) < 3:
            top_results.append(res)
            top_results.sort(key=lambda x: x['delta_e'])
        else:
            if res['delta_e'] < top_results[-1]['delta_e']:
                top_results[-1] = res
                top_results.sort(key=lambda x: x['delta_e'])
    return top_results

# ---------- Точные совпадения ----------
def find_exact_matches(target_lab, paints_df):
    matches = []
    target_L, target_a, target_b = target_lab
    for _, row in paints_df.iterrows():
        rgb = (int(row['hex'][0:2], 16), int(row['hex'][2:4], 16), int(row['hex'][4:6], 16))
        L, a, b = rgb_to_lab_fast(rgb)
        delta = delta_e_cie2000_fast(target_L, target_a, target_b, L, a, b)
        if delta <= EPSILON:
            matches.append({'paint': row, 'delta_e': delta})
    return matches

def delta_to_percent(delta):
    return max(0.0, min(100.0, 100.0 - delta * 10.0))

# ---------- Формирование альтернатив (без дублирования) ----------
def build_hex_alternatives(df):
    alt = {}
    for _, row in df.iterrows():
        h = row['hex']
        pair = (row['article'], row['brand'])
        if h not in alt:
            alt[h] = set()
        alt[h].add(pair)
    return {h: [{'article': art, 'brand': br} for art, br in pairs] for h, pairs in alt.items() if len(pairs) > 1}

def format_alternatives(hex_code, alt_dict, current_article=None, current_brand=None):
    if hex_code in alt_dict:
        alts = alt_dict[hex_code]
        filtered = [a for a in alts if (a['article'], a['brand']) != (current_article, current_brand)]
        if not filtered:
            return ""
        parts = [f"арт.{a['article']}, {a['brand']}" for a in filtered]
        return " (альт.: " + "; ".join(parts) + ")"
    return ""

def format_paint_with_alternatives(paint_row, alt_dict):
    hex_code = paint_row['hex']
    block = f"[{hex_code}]"
    base = f"{block} {paint_row['brand']} / {paint_row['name']} (арт.{paint_row['article']})"
    alt = format_alternatives(hex_code, alt_dict, paint_row['article'], paint_row['brand'])
    return base + alt

# ---------- Вывод результатов ----------
def print_results(target_hex, result_dict, alt_dict):
    print("\n" + "="*70)
    print(f"Целевой цвет: #{target_hex.upper()}")
    print("="*70)
    if result_dict['type'] == 'exact':
        matches = result_dict['result']
        print(f"\n[OK] Точное совпадение (Delta E ≤ {EPSILON}):")
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
    print(f"\n Лучшие смеси из {n} красок:")
    for i, mix in enumerate(top):
        real_perc = delta_to_percent(mix['delta_e'])
        print(f"\nВариант {i+1}: совпадение ~{real_perc:.1f}%")
        # Вычисляем части для каждой краски
        weights = mix['weights']
        fracs = [Fraction(w).limit_denominator(8) for w in weights]
        denoms = [f.denominator for f in fracs]
        common_denom = 1
        for d in denoms:
            common_denom = common_denom * d // gcd(common_denom, d)
        parts_list = [f.numerator * (common_denom // f.denominator) for f in fracs]
        g = parts_list[0]
        for p in parts_list[1:]:
            g = gcd(g, p)
        parts_list = [p // g for p in parts_list]
        print("Исходные краски:")
        for j, p in enumerate(mix['paints']):
            part = parts_list[j]
            percent = weights[j] * 100
            # Склонение
            if part % 10 == 1 and part % 100 != 11:
                part_word = f"{part} часть"
            elif 2 <= part % 10 <= 4 and (part % 100 < 10 or part % 100 >= 20):
                part_word = f"{part} части"
            else:
                part_word = f"{part} частей"
            hex_code = p['hex']
            block = f"[{hex_code}]"
            desc = f"{p['brand']} / {p['name']} (арт.{p['article']})"
            print(f"{part_word} ({percent:.1f}%) - {block} {desc}")
        # Результирующий цвет
        mixed_lab = mix['mixed_lab']
        mixed_rgb = lab_to_rgb_fast(mixed_lab)
        mixed_hex = f"{mixed_rgb[0]:02X}{mixed_rgb[1]:02X}{mixed_rgb[2]:02X}"
        mixed_block = f"[{mixed_hex}]"
        print(f"  Результат смешивания: {mixed_block}")
#    print("\n* 100% = Delta E=0 (идеал), 0% = Delta E≥10")

# ---------- Основной цикл с интерактивным предложением ----------
def interactive():
    print(" ПОДБОР СМЕСИ КРАСОК")
    print("="*70)
    if USE_NUMBA:
        print("[*] Ускорение: Numba JIT включено")
    else:
        print("[!] Numba не установлена, установите для ускорения: pip install numba")

    paints = load_paints()
    alt_dict = build_hex_alternatives(paints)
    if alt_dict:
        print(f"[i] Обнаружены дубликаты по HEX: {len(alt_dict)} цветов имеют альтернативы.")

    while True:
        target = input("\nЦелевой HEX (например FFD700) или Enter/q для выхода: ").strip().lstrip('#')
        if target.lower() in ('', 'q', 'exit', 'quit'):
            print("Выход.")
            break
        if len(target) != 6 or not all(c in '0123456789ABCDEFabcdef' for c in target):
            print("Неверный HEX. Нужно 6 символов (0-9, A-F).")
            continue
        target = target.upper()
        target_rgb = (int(target[0:2], 16), int(target[2:4], 16), int(target[4:6], 16))
        target_lab = rgb_to_lab_fast(target_rgb)

        # 1. Точные совпадения
        exact = find_exact_matches(target_lab, paints)
        if exact:
            print_results(target, {'type': 'exact', 'result': exact}, alt_dict)
            continue

        # 2. Поиск смесей из 2 красок
        best_2 = find_best_mix_2(target_lab, paints, top_k=200)
        if best_2:
            best_percent = delta_to_percent(best_2[0]['delta_e'])
            print_results(target, {'type': 'mix2', 'result': best_2}, alt_dict)

            # Если лучший вариант < 95% – предложить проверить 3 краски
            if best_percent < 95.0:
                answer = input("\nСовпадение менее 95%. Хотите проверить смеси из 3 красок? (y/n): ").strip().lower()
                if answer == 'y':
                    best_3 = find_best_mix_3(target_lab, paints, top_k=100)
                    if best_3:
                        print_results(target, {'type': 'mix3', 'result': best_3}, alt_dict)
                    else:
                        print("Не найдено подходящих смесей из 3 красок.")
        else:
            # Если нет ни одной пары – сразу пробуем тройки
            print("Не найдено подходящих смесей из 2 красок. Пробуем 3...")
            best_3 = find_best_mix_3(target_lab, paints, top_k=100)
            if best_3:
                print_results(target, {'type': 'mix3', 'result': best_3}, alt_dict)
            else:
                print("Не найдено подходящих смесей.")

if __name__ == '__main__':
    mp.freeze_support()
    try:
        interactive()
    except Exception as e:
        print("Ошибка:", e, file=sys.stderr)
        traceback.print_exc()
        input("Нажмите Enter для выхода...")