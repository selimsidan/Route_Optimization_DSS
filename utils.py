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
def compute_route_distance(route, dist_matrix):
    """Verilen rota (indeks listesi) için toplam mesafeyi hesaplar.
       Rota; başlangıç ve bitiş depot’u içerir."""
    total = 0.0
    for i in range(len(route)-1):
        total += dist_matrix[route[i]][route[i+1]]
    return total

def two_opt(route, dist_matrix):
    """İki‑opt yerelleştirme algoritması."""
    best_route = route.copy()
    best_distance = compute_route_distance(best_route, dist_matrix)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route)-2):
            for j in range(i+1, len(best_route)-1):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_distance = compute_route_distance(new_route, dist_matrix)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
    return best_route

def three_opt(route, dist_matrix):
    """Çok basit 3‑opt yerelleştirme; kısıtlı kombinasyon deneniyor."""
    best_route = route.copy()
    best_distance = compute_route_distance(best_route, dist_matrix)
    improved = True
    n = len(route)
    while improved:
        improved = False
        for i in range(1, n-4):
            for j in range(i+1, n-2):
                for k in range(j+1, n):
                    segments = [best_route[:i], best_route[i:j], best_route[j:k], best_route[k:]]
                    candidate = segments[0] + segments[1][::-1] + segments[2][::-1] + segments[3]
                    candidate_distance = compute_route_distance(candidate, dist_matrix)
                    if candidate_distance < best_distance:
                        best_route = candidate
                        best_distance = candidate_distance
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return best_route

def simulated_annealing(initial_route, dist_matrix, initial_temp=1000, cooling_rate=0.99, stopping_temp=1, max_restarts=100, patience=1000):
    def get_route_distance(route):
        return compute_route_distance(route, dist_matrix)

    def get_neighbor(route):
        # Preserve the start and end nodes (depots)
        new_route = route.copy()
        start, end = new_route[0], new_route[-1]
        intermediate_nodes = new_route[1:-1]
        
        # Randomly choose between 2-opt and 3-opt
        if random.random() < 0.5:
            # 2-opt move
            i, j = sorted(random.sample(range(len(intermediate_nodes)), 2))
            intermediate_nodes[i:j+1] = reversed(intermediate_nodes[i:j+1])
        else:
            # 3-opt move
            if len(intermediate_nodes) < 3:
                return new_route  # Not enough nodes for 3-opt
            i, j, k = sorted(random.sample(range(len(intermediate_nodes)), 3))
            segments = [intermediate_nodes[:i], intermediate_nodes[i:j], intermediate_nodes[j:k], intermediate_nodes[k:]]
            new_route = [start] + segments[0] + segments[1][::-1] + segments[2][::-1] + segments[3] + [end]
            return new_route

        new_route = [start] + intermediate_nodes + [end]
        return new_route

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

            if delta_distance < 0 or random.random() < math.exp(-delta_distance / temperature):
                current_route = neighbor_route
                current_distance = neighbor_distance
                no_improvement_count = 0

                if current_distance < best_distance:
                    best_route = current_route
                    best_distance = current_distance
            else:
                no_improvement_count += 1

            # Early stop if no improvement
            if no_improvement_count > patience:
                break

            temperature *= cooling_rate

        # Track the best overall solution across restarts
        if best_distance < best_overall_distance:
            best_overall_route = best_route
            best_overall_distance = best_distance

        # Shake up the initial route if no improvement
        if best_overall_distance == get_route_distance(initial_route):
            random.shuffle(initial_route[1:-1])  # Shuffle only intermediate nodes

    return best_overall_route, best_overall_distance