import streamlit as st
import pandas as pd
import numpy as np
import pulp
import folium
from streamlit_folium import folium_static
from geopy import distance
import requests
import polyline
import random
from itertools import combinations

class AdvancedVRPSolver:
    def __init__(self, data, num_vehicles, vehicle_capacity, max_route_time):
        self.data = data
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.max_route_time = max_route_time
        
        # ID kolonunu güvenli bir şekilde oluştur
        if 'ID' not in data.columns:
            data['ID'] = data.index
        
        self.coordinates = data[['Latitude', 'Longitude']].values
        self.demands = data['demand'].values
        self.distances = self._calculate_distance_matrix()
        self.depot_index = 0
        
        # Ziyaret edilmemiş nodeları takip etmek için
        self.unvisited_nodes = set(range(1, len(data)))

    def _calculate_distance_matrix(self):
        """Gelişmiş coğrafi mesafe matrisi hesaplama"""
        n = len(self.coordinates)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i][j] = distance.distance(
                        self.coordinates[i], 
                        self.coordinates[j]
                    ).kilometers
        
        return distance_matrix

    def _apply_clarke_wright_savings(self, routes):
        """Clarke-Wright Savings algoritması ile rota optimize etme"""
        savings = []
        for i, j in combinations(range(1, len(self.data)), 2):
            saving = (self.distances[self.depot_index][i] + 
                      self.distances[self.depot_index][j] - 
                      self.distances[i][j])
            savings.append((saving, i, j))
        
        savings.sort(reverse=True)
        
        for saving, i, j in savings:
            # Rota optimizasyon kontrolü
            route_i = next((r for r in routes if i in r), None)
            route_j = next((r for r in routes if j in r), None)
            
            if route_i and route_j and route_i != route_j:
                # Kapasite kontrolü
                total_demand = sum(self.demands[n] for n in route_i + route_j)
                if total_demand <= self.vehicle_capacity:
                    routes.remove(route_i)
                    routes.remove(route_j)
                    merged_route = self._merge_routes(route_i, route_j)
                    routes.append(merged_route)
        
        return routes

    def _merge_routes(self, route1, route2):
        """Rotaları akıllıca birleştirme"""
        if route1[1] == route2[-2]:
            return route1[:-1] + route2[1:]
        elif route1[-2] == route2[1]:
            return route2[:-1] + route1[1:]
        elif route1[1] == route2[1]:
            return route1[:-1] + route2[::-1][1:]
        elif route1[-2] == route2[-2]:
            return route1 + route2[::-1][1:]

    def solve_vrp_milp(self, time_limit=None):
        """Gelişmiş MILP VRP Çözücü"""
        prob = pulp.LpProblem("Vehicle_Routing_Problem", pulp.LpMinimize)
        n = len(self.data)
        
        # Gelişmiş karar değişkenleri
        x = {}
        for i in range(n):
            for j in range(n):
                for k in range(self.num_vehicles):
                    x[i, j, k] = pulp.LpVariable(f'x_{i}_{j}_{k}', cat='Binary')
        
        u = {}
        for i in range(1, n):
            for k in range(self.num_vehicles):
                u[i, k] = pulp.LpVariable(f'u_{i}_{k}', lowBound=0, upBound=self.vehicle_capacity)
        
        # Objektif fonksiyon: Toplam mesafeyi minimize et
        prob += pulp.lpSum(
            self.distances[i][j] * x[i, j, k]
            for i in range(n)
            for j in range(n)
            for k in range(self.num_vehicles) if i != j
        )
        
        # Gelişmiş kısıtlamalar
        # 1. Her müşteri tam olarak bir kez ziyaret edilmeli
        for j in range(1, n):
            prob += pulp.lpSum(
                x[i, j, k] for i in range(n) 
                for k in range(self.num_vehicles) if i != j
            ) == 1
        
        # 2. Her araç depoya çıkmalı ve geri dönmeli
        for k in range(self.num_vehicles):
            prob += pulp.lpSum(x[self.depot_index, j, k] for j in range(1, n)) <= 1
            prob += pulp.lpSum(x[j, self.depot_index, k] for j in range(1, n)) <= 1
        
        # 3. Akış koruması
        for k in range(self.num_vehicles):
            for i in range(1, n):
                prob += pulp.lpSum(x[j, i, k] for j in range(n) if j != i) == \
                        pulp.lpSum(x[i, j, k] for j in range(n) if j != i)
        
        # 4. Araç kapasite kısıtı
        for k in range(self.num_vehicles):
            for i in range(1, n):
                for j in range(1, n):
                    if i != j:
                        prob += u[i, k] + self.demands[j] <= \
                                u[j, k] + self.vehicle_capacity * (1 - x[i, j, k])
        
        # 5. Rota zaman kısıtı
        speed_kmh = 50  # Ortalama araç hızı
        time_matrix = self.distances / speed_kmh
        for k in range(self.num_vehicles):
            prob += pulp.lpSum(
                time_matrix[i][j] * x[i, j, k]
                for i in range(n)
                for j in range(n) if i != j
            ) <= self.max_route_time
        
        # Çözümü başlat
        best_solution_value = float('inf')  # En iyi çözüm değerini başlangıçta çok büyük olarak ayarla
        best_routes = None  # En iyi rotaları saklamak için
        
        # Eğer zaman limiti belirlenmişse, çözümlemeden önce timeLimit'i ayarla
        if time_limit:
            prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit))
        else:
            prob.solve()
        
        # En iyi çözümü ilk başta kontrol et
        if pulp.LpStatus[prob.status] == "Optimal":
            best_solution_value = pulp.value(prob.objective)
            best_routes = self._extract_routes_from_solution(prob, n)
        
        # Çözüm limitine ulaşılırsa, önceki çözümü kullan
        if time_limit and pulp.LpStatus[prob.status] != "Optimal":
            st.warning("Zaman Limiti Aşıldı! Bulunan en iyi çözüm kullanılıyor.")
            best_routes = self._extract_routes_from_solution(prob, n)
        
        return best_routes, best_solution_value

    def _extract_routes_from_solution(self, prob, n):
        """Optimal çözümden rotaları çıkarma"""
        routes = [[] for _ in range(self.num_vehicles)]
        self.unvisited_nodes = set(range(1, n))
        
        for k in range(self.num_vehicles):
            current = self.depot_index
            route = [current]
            
            while True:
                next_node = None
                for j in range(n):
                    if j != current and j in self.unvisited_nodes and pulp.value(prob.variablesDict()[f'x_{current}_{j}_{k}']) == 1:
                        next_node = j
                        break
                
                if next_node is None:
                    route.append(self.depot_index)
                    break
                
                route.append(next_node)
                self.unvisited_nodes.remove(next_node)
                current = next_node
            
            if len(route) > 2:
                routes[k] = route
                
        return routes



    def create_advanced_route_map(self, routes):
        """Gelişmiş rota haritası oluşturma"""
        center_lat = self.data['Latitude'].mean()
        center_lon = self.data['Longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        colors = [
            'red', 'blue', 'green', 'purple', 'orange', 
            'darkred', 'darkblue', 'cadetblue', 'darkgreen', 
            'darkpurple', 'pink', 'lightred', 'beige'
        ]
        
        # Depo marker
        depot_row = self.data.iloc[self.depot_index]
        folium.Marker(
            [depot_row['Latitude'], depot_row['Longitude']],
            popup="Depo",
            icon=folium.Icon(color='black', icon='home')
        ).add_to(m)
        
        for idx, row in self.data.iterrows():
            if idx == self.depot_index:
                continue
            marker_color = 'gray' if idx in self.unvisited_nodes else 'green'
            icon_type = 'exclamation' if idx in self.unvisited_nodes else 'info-sign'
            
            folium.Marker(
                [row['Latitude'], row['Longitude']],
                popup=f"Müşteri {row['ID']} - Talep: {row['demand']}",
                icon=folium.Icon(color=marker_color, icon=icon_type)
            ).add_to(m)
        
        for vehicle, route in enumerate(routes):
            if not route:
                continue
            
            color = colors[vehicle % len(colors)]
            route_coords = []
            
            for i in range(len(route) - 1):
                from_node, to_node = route[i], route[i+1]
                from_data = self.data.iloc[from_node]
                to_data = self.data.iloc[to_node]
                
                route_coords.append((from_data['Latitude'], from_data['Longitude']))
                route_coords.append((to_data['Latitude'], to_data['Longitude']))
            
            if route_coords:
                folium.PolyLine(
                    route_coords,
                    color=color,
                    weight=4,
                    opacity=0.7,
                    popup=f"Araç {vehicle + 1} Rotası"
                ).add_to(m)
        
        return m

# Streamlit UI
st.set_page_config(page_title="Gelişmiş VRP Çözümleme", layout="wide")
st.title("🚚 Gelişmiş Araç Rotalama Problemi Çözücü")

with st.sidebar:
    st.header("Rotalama Parametreleri")
    vehicles = st.number_input("Araç Sayısı:", min_value=1, value=4, step=1)
    vehicle_capacity = st.number_input("Araç Kapasitesi:", min_value=1, value=200, step=1)
    max_route_time = st.number_input("Maksimum Rota Süresi (saat):", min_value=1, value=8, step=1)
    time_limit = st.number_input("Zaman Limiti (saniye):", min_value=1, value=600, step=1)

uploaded_file = st.file_uploader("Excel Dosyası Yükle", type="xlsx")

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    
    required_columns = ['Latitude', 'Longitude', 'demand']
    if not all(col in data.columns for col in required_columns):
        st.error("Geçersiz Excel dosyası. Gerekli kolonlar: Latitude, Longitude, demand")
    else:
        vrp_solver = AdvancedVRPSolver(
            data, 
            num_vehicles=vehicles, 
            vehicle_capacity=vehicle_capacity,
            max_route_time=max_route_time
        )
        
        routes, total_distance = vrp_solver.solve_vrp_milp(time_limit=time_limit)
        
        # Sonuç gösterimi
        st.header("Rotalama Sonuçları")
        st.metric("Toplam Rota Mesafesi", f"{total_distance:.2f} km")
        
        # Ziyaret edilmemiş nodların bilgisi
        if vrp_solver.unvisited_nodes:
            st.warning("Ziyaret Edilmemiş Müşteri Noktaları:")
            unvisited_ids = [vrp_solver.data.iloc[node]['ID'] for node in vrp_solver.unvisited_nodes]
            st.write(unvisited_ids)
        
        # Rota detayları
        with st.expander("Rota Detayları"):
            for i, route in enumerate(routes, 1):
                if route:
                    route_ids = [vrp_solver.data.iloc[node]['ID'] for node in route]
                    st.write(f"Araç {i}: Müşteri ID'leri {route_ids}")
        
        # Rota görselleştirmesi
        st.header("Rota Görselleştirmesi")
        route_map = vrp_solver.create_advanced_route_map(routes)
        folium_static(route_map, width=1000, height=600)

else:
    st.warning("Lütfen lokasyon verilerini içeren bir Excel dosyası yükleyiniz.")
