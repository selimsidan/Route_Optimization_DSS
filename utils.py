import streamlit as st
import pandas as pd
import numpy as np
import pulp
import folium
from streamlit_folium import st_folium, folium_static
from geopy import distance
import math
import requests
import polyline
from sklearn.cluster import KMeans
import random

###############################################
# Yardımcı Fonksiyonlar: 2‑opt, 3‑opt ve rota mesafesi hesaplama
###############################################
import math

def compute_route_distance(route, dist_matrix):
    """
    Given a route (list of node indices) and a distance matrix,
    returns the total distance of the route.

    route: e.g. [0, 3, 5, 1, 0]
    dist_matrix: a 2D list or numpy array where dist_matrix[i][j]
                 is the distance from node i to node j.
    """
    total_dist = 0.0
    for i in range(len(route) - 1):
        total_dist += dist_matrix[route[i]][route[i+1]]
    return total_dist

def two_opt(route, dist_matrix):
    """
    Simple 2-opt local search for a single route.
    Tries reversing every possible sub-path to see if it improves distance.
    Returns the improved route.
    """
    best_route = route[:]
    best_distance = compute_route_distance(best_route, dist_matrix)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                # Create a candidate by reversing the segment [i:j+1]
                candidate = (best_route[:i]
                             + best_route[i:j+1][::-1]
                             + best_route[j+1:])
                candidate_distance = compute_route_distance(candidate, dist_matrix)
                if candidate_distance < best_distance:
                    best_route = candidate
                    best_distance = candidate_distance
                    improved = True
                    break
            if improved:
                break
    return best_route

def all_3_opt_moves(route, i, j, k):
    """
    Given a route and three cut points (i, j, k),
    returns all 8 possible 3-opt reconnection patterns.

    A, B, C, D are segments:
      route = A + B + C + D
    i, j, k: indices where we cut.
    """
    A = route[:i]
    B = route[i:j]
    C = route[j:k]
    D = route[k:]

    return [
        A + B + C + D,                # 1) Original (no change)
        A + B[::-1] + C[::-1] + D,    # 2) Reverse B, reverse C
        A + C[::-1] + B[::-1] + D,    # 3) Swap & reverse B, C
        A + C + B + D,                # 4) Swap B, C
        A + B[::-1] + C + D,          # 5) Reverse B only
        A + B + C[::-1] + D,          # 6) Reverse C only
        A + C + B[::-1] + D,          # 7) Swap B, C + reverse B
        A + C[::-1] + B + D           # 8) Swap B, C + reverse C
    ]

def three_opt(route, dist_matrix):
    """
    Performs a classical (fully enumerated) 3-opt local search on a single route.
    Returns an improved route if found.

    - This is a repeated local search:
      We keep trying 3-opt moves until no improvement is found.
    - For each triple (i, j, k), we generate all possible reconnections
      and accept the first improvement (first-improvement strategy).
    """
    best_route = route[:]
    best_distance = compute_route_distance(best_route, dist_matrix)
    improved = True

    while improved:
        improved = False
        # Loop over all triples (i, j, k)
        # Typically skipping the last index if it's a depot, etc.
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                for k in range(j + 1, len(best_route)):
                    new_routes = all_3_opt_moves(best_route, i, j, k)
                    for candidate in new_routes:
                        candidate_distance = compute_route_distance(candidate, dist_matrix)
                        if candidate_distance < best_distance:
                            # Accept improvement immediately (first-improvement)
                            best_route = candidate
                            best_distance = candidate_distance
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    return best_route

def improve_all_routes(all_routes, dist_matrix, max_iterations=5):
    """
    Repeatedly apply 3-opt (or 2-opt) to each route in the VRP.
    If a route is very short, do 2-opt or skip.
    all_routes: dict of { vehicle_id: route_list }
    """
    for _ in range(max_iterations):
        for vehicle_id, route in all_routes.items():
            # If route is too short (fewer than 5 nodes), 3-opt isn't possible
            # because we can't pick three distinct cut points.
            if len(route) < 5:
                improved_route = two_opt(route, dist_matrix)
            else:
                improved_route = three_opt(route, dist_matrix)
            all_routes[vehicle_id] = improved_route
    return all_routes

def total_distance_of_all_routes(all_routes, dist_matrix):
    """
    Sums the distance of each route in all_routes.
    all_routes: dict of { vehicle_id: route_list }
    """
    total = 0.0
    for vehicle_id, route in all_routes.items():
        total += compute_route_distance(route, dist_matrix)
    return total


def simulated_annealing(initial_route, dist_matrix, 
                        initial_temp=1000, cooling_rate=0.99, stopping_temp=1, 
                        max_restarts=20, patience=1000):
    """
    Simulated Annealing algoritması.
    - Kısa rotalar (len < 5) için 2-opt neighbor,
    - Daha uzun rotalar (len >= 5) için 3-opt neighbor kullanır.
    """

    def compute_route_distance(route, dist_matrix):
        """Verilen rota (indeks listesi) için toplam mesafeyi hesaplar."""
        total = 0.0
        for i in range(len(route)-1):
            total += dist_matrix[route[i]][route[i+1]]
        return total

    def get_route_distance(route):
        return compute_route_distance(route, dist_matrix)

    def two_opt_neighbor(route):
        """2-opt komşu üretimi."""
        new_route = route.copy()
        # Başlangıç (0) ve bitiş (son) depo indekslerini sabit tutmak istediğimizi varsayıyoruz:
        i, j = sorted(random.sample(range(1, len(route) - 1), 2))
        new_route[i:j+1] = reversed(new_route[i:j+1])
        return new_route

    def three_opt_neighbor(route):
        """3-opt komşu üretimi."""
        new_route = route.copy()
        # 3 farklı indeks seçiyoruz (depo dışı):
        i, j, k = sorted(random.sample(range(1, len(route) - 1), 3))

        A = new_route[:i]
        B = new_route[i:j]
        C = new_route[j:k]
        D = new_route[k:]

        candidates = []
        # Farklı 3‑opt kombinasyonları:
        candidates.append(A + B[::-1] + C[::-1] + D)  # Option 1
        candidates.append(A + B[::-1] + C + D)        # Option 2
        candidates.append(A + B + C[::-1] + D)        # Option 3
        candidates.append(A + C + B + D)              # Option 4
        candidates.append(A + (B + C)[::-1] + D)      # Option 5

        # Rastgele birini seç:
        return random.choice(candidates)

    def get_neighbor(route):
        """
        Rota kısa ise (4 node veya daha az), 2-opt;
        yeterince uzun ise 3-opt kullanarak komşu üret.
        """
        if len(route) < 5:
            return two_opt_neighbor(route)
        else:
            return three_opt_neighbor(route)

    # Başlangıç
    best_overall_route = initial_route
    best_overall_distance = get_route_distance(initial_route)

    for restart in range(max_restarts):
        current_route = initial_route
        current_distance = get_route_distance(current_route)
        best_route = current_route
        best_distance = current_distance
        temperature = initial_temp
        no_improvement_count = 0

        while temperature > stopping_temp:
            neighbor_route = get_neighbor(current_route)
            neighbor_distance = get_route_distance(neighbor_route)
            delta_distance = neighbor_distance - current_distance

            # Kabul kriteri
            if delta_distance < 0 or random.random() < math.exp(-delta_distance / temperature):
                current_route = neighbor_route
                current_distance = neighbor_distance
                no_improvement_count = 0

                # En iyi lokal çözüm
                if current_distance < best_distance:
                    best_route = current_route
                    best_distance = current_distance
            else:
                no_improvement_count += 1

            # Gelişme yoksa erken çıkış
            if no_improvement_count > patience:
                break

            # Soğuma
            temperature *= cooling_rate

        # Tüm restartlar arasında en iyi çözüme bak
        if best_distance < best_overall_distance:
            best_overall_route = best_route
            best_overall_distance = best_distance

        # Hiç iyileşme olmadıysa, başlangıç rotasını karıştır
        if best_overall_distance == get_route_distance(initial_route):
            random.shuffle(initial_route)

    return best_overall_route, best_overall_distance