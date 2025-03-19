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
import time  # Progress bar için gerekli
import base64  # Örnek dosya indirmek için gerekli
import io

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

def simulated_annealing(initial_route, dist_matrix, initial_temp=1000, cooling_rate=0.995, stopping_temp=1):
    def get_route_distance(route):
        return compute_route_distance(route, dist_matrix)
    def get_neighbor(route):
        new_route = route.copy()
        i, j = sorted(random.sample(range(1, len(route) - 1), 2))
        new_route[i:j+1] = reversed(new_route[i:j+1])
        return new_route

    current_route = initial_route
    current_distance = get_route_distance(current_route)
    best_route = current_route
    best_distance = current_distance
    temperature = initial_temp

    while temperature > stopping_temp:
        neighbor_route = get_neighbor(current_route)
        neighbor_distance = get_route_distance(neighbor_route)
        delta_distance = neighbor_distance - current_distance

        if delta_distance < 0 or random.random() < math.exp(-delta_distance / temperature):
            current_route = neighbor_route
            current_distance = neighbor_distance

            if current_distance < best_distance:
                best_route = current_route
                best_distance = current_distance

        temperature *= cooling_rate

    return best_route, best_distance

###############################################
# Gelişmiş VRP Çözücü – MILP ve Heuristic (Çoklu Depo, Locker Ataması, Forbidden Node Kısıtı, OSRM)
###############################################
class AdvancedVRPSolver:
    def __init__(self, data):
        """
        data: VRP’ye sokulacak (işlenmiş) DataFrame.
              – DataFrame’de node_type sütunu: depot, locker, customer şeklinde tanımlı.
              – deliver_type bilgisi bulunmamakta; model karar değişkenleri üzerinden servis şekli belirlenecektir.
        """
        self.data = data.reset_index(drop=True)
        if 'ID' not in self.data.columns:
            self.data['ID'] = self.data.index
        # Eğer cost bilgisi yoksa varsayılan olarak 1.0 değeri atanır.
        if 'cost' not in self.data.columns:
            self.data['cost'] = 1.0
        # Depo indekslerini belirle:
        self.depot_indices = self.data.index[self.data['node_type'] == 'depot'].tolist()
        if not self.depot_indices:
            raise ValueError("En az bir depo bulunmalıdır!")
        self.coordinates = self.data[['Latitude','Longitude']].values
        # Customer'ların demand değeri, locker'lar için 0 kabul ediliyor.
        self.demands = self.data.apply(lambda row: row['demand'] if row['node_type'] != 'locker' else 0, axis=1).values
        self.dist_matrix = self._calculate_distance_matrix()
        # Customer indekslerini belirleyelim:
        self.customer_indices = self.data.index[self.data['node_type'] == 'customer'].tolist()
        # Locker indekslerini belirleyelim:
        self.locker_indices = self.data.index[self.data['node_type'] == 'locker'].tolist()

    def _calculate_distance_matrix(self):
        n = len(self.coordinates)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = distance.distance(self.coordinates[i], self.coordinates[j]).kilometers
        return matrix

    def get_osrm_route(self, start_coord, end_coord, cache):
        key = (start_coord, end_coord)
        if key in cache:
            return cache[key]
        url = f"http://router.project-osrm.org/route/v1/driving/{start_coord[1]},{start_coord[0]};{end_coord[1]},{end_coord[0]}?overview=full&geometries=polyline"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 'Ok' and data.get('routes'):
                    polyline_str = data['routes'][0]['geometry']
                    coords = polyline.decode(polyline_str)
                    cache[key] = coords
                    return coords
            cache[key] = [start_coord, end_coord]
            return [start_coord, end_coord]
        except Exception as e:
            cache[key] = [start_coord, end_coord]
            return [start_coord, end_coord]

    def violates_forbidden(self, route, candidate, forbidden_groups):
        candidate_id = self.data.loc[candidate, 'ID']
        route_ids = [self.data.loc[node, 'ID'] for node in route]
        for group in forbidden_groups:
            if candidate_id in group:
                if any(rid in group for rid in route_ids):
                    return True
        return False

    def fix_route_forbidden(self, route, forbidden_groups):
        new_route = [route[0]]
        for node in route[1:-1]:
            candidate_id = self.data.loc[node, 'ID']
            conflict = False
            for group in forbidden_groups:
                if candidate_id in group and any(self.data.loc[n, 'ID'] in group for n in new_route):
                    conflict = True
                    break
            new_route.append(node)
        new_route.append(route[-1])
        return new_route

    ##########################################################
    # MILP Tabanlı Çözüm
    ##########################################################
    def solve_vrp_milp(self, vehicles_df, forbidden_groups=None, time_limit=None, locker_km_limit=10):
        """
        Gelişmiş MILP tabanlı VRP çözümü:
        - Tek depo, çok locker
        - Müşteriler: Tüm 'customer' tipindeki node'lar, eğer locker seçeneği varsa 
          home[i] + Σ(z[i,lk]) = 1 kısıtı ile ya evine teslimat (home[i]=1) ya da locker ataması (z[i,lk]=1) yapılır.
        - Locker'lar: Kendisine atanan toplam talep d[j], kullanım durumu w[j], rota üzerinde araç yük drop-off modellemesi.
        - Rota: depo çıkış/varış, akış (flow), araç kapasitesi ve zaman (maksimum çalışma süresi) kısıtları.
        - Yük modellemesi: r[i,v] => araç v, i düğümünden ayrılırken elindeki kalan yük. 
          (Eğer j customer ise demands[j], j locker ise d[j] kadar yük drop-off yapılır.)
        - Yalnızca ev teslimatı yapılan customers ve atanan lockerlar rota üzerinde ziyaret edilir.
        - Modelin sonunda hangi customer hangi lockera atandı, ne kadar atama maliyeti oluştu vb. veriler de elde edilir.
        - NOT: w[j] >= z[i,j] kısıtları eklenmiştir; eğer herhangi bir müşteri i locker j'ye atanmışsa, w[j] = 1 zorunludur.
        """
        if forbidden_groups is None:
            forbidden_groups = []
        
        # 1) Araç Parametreleri
        vehicle_ids = list(vehicles_df['vehicle_id'])
        vehicles = {}
        for _, row in vehicles_df.iterrows():
            vehicles[row['vehicle_id']] = {
                'capacity': row['capacity'],
                'max_duration': row['max_duration'] / 60.0,  # dakika -> saat
                'cost_per_km': row['cost_per_km'],
                'fixed_cost': row['fixed_cost']
            }
        
        # 2) Node Kümeleri
        N = list(range(self.data.shape[0]))  # Tüm düğüm indeksleri
        D = [i for i in N if str(self.data.loc[i, 'node_type']).strip().lower() == 'depot']
        L = [i for i in N if str(self.data.loc[i, 'node_type']).strip().lower() == 'locker']
        M = [i for i in N if str(self.data.loc[i, 'node_type']).strip().lower() not in ['depot', 'locker']]
        
        # Her müşteri için locker aday kümesi
        candidate_lockers = {}
        for i in M:
            candidate_lockers[i] = [lk for lk in L if self.dist_matrix[i][lk] <= locker_km_limit]
        
        # 3) Model Oluşturma
        prob = pulp.LpProblem("SingleDepot_VRP_CustomerLocker", pulp.LpMinimize)
        
        # 4) Karar Değişkenleri: x[i,j,v] = 1 => araç v, i'den j'ye gidiyor
        x = {}
        for i in N:
            for j in N:
                if i == j:
                    continue
                for v in vehicle_ids:
                    x[i, j, v] = pulp.LpVariable(f"x_{i}_{j}_{v}", cat="Binary")
        
        # y[v] => araç v kullanılıyor mu
        y = {v: pulp.LpVariable(f"y_{v}", cat="Binary") for v in vehicle_ids}
        
        # Müşteri atama: home[i], z[i,j]
        home = {i: pulp.LpVariable(f"home_{i}", cat="Binary") for i in M}
        z = {}
        for i in M:
            if candidate_lockers[i]:
                for j in candidate_lockers[i]:
                    z[i, j] = pulp.LpVariable(f"z_{i}_{j}", cat="Binary")
                prob += home[i] + pulp.lpSum(z[i, j] for j in candidate_lockers[i]) == 1, f"Assign_{i}"
            else:
                # Uygun locker yoksa home delivery zorunlu
                prob += home[i] == 1, f"ForceHome_{i}"
        
        # Locker kullanım: w[j], d[j]
        w = {j: pulp.LpVariable(f"w_{j}", cat="Binary") for j in L}
        d_locker = {}
        for j in L:
            d_locker[j] = pulp.LpVariable(f"d_{j}", lowBound=0)
            candidate_customers = [i for i in M if (i, j) in z]
            if candidate_customers:
                prob += d_locker[j] == pulp.lpSum(self.demands[i] * z[i, j] for i in candidate_customers), f"DemandAssign_{j}"
            else:
                prob += d_locker[j] == 0, f"DemandAssign_{j}"
            
            locker_cap = self.data.loc[j, 'capacity']
            prob += d_locker[j] <= locker_cap, f"LockerCap_{j}"
            prob += d_locker[j] <= locker_cap * w[j], f"LockerUsage_{j}"
            
            # Burada "sum z[i,j] >= w[j]" yerine "sum z[i,j] <= M_count * w[j]" da ekleyebilirsiniz.
            # Veya "w[j] >= z[i,j]" tek tek eklemek de bir seçenek:
            for i_cust in candidate_customers:
                # eğer i_cust bu lockera atanmışsa (z[i_cust,j]=1), w[j] de 1 olmak zorunda
                prob += w[j] >= z[i_cust, j], f"LinkCustLocker_{i_cust}_{j}"
        
        # Rota'ya dahil M ve L
        M_route = M + L
        
        # 8) Araçta Kalan Yük Değişkeni: r[i,v]
        r = {}
        for i in M_route:
            for v in vehicle_ids:
                r[i, v] = pulp.LpVariable(f"r_{i}_{v}", lowBound=0, upBound=vehicles[v]['capacity'])
        
        # (A) MTZ benzeri kısıt: i->j geçişinde yük azalması
        for v in vehicle_ids:
            for i in M_route:
                for j in M_route:
                    if i == j:
                        continue
                    if j in M:  # j customer
                        prob += (r[i, v] - r[j, v] >=
                                 self.demands[j] - vehicles[v]['capacity'] * (1 - x[i, j, v])), f"MTZ_customer_{i}_{j}_{v}"
                    else:       # j locker
                        prob += (r[i, v] - r[j, v] >=
                                 d_locker[j] - vehicles[v]['capacity'] * (1 - x[i, j, v])), f"MTZ_locker_{i}_{j}_{v}"
        
        # 9) Delta[v,i]
        delta = {}
        for v in vehicle_ids:
            for i in M_route:
                delta[v, i] = pulp.LpVariable(f"delta_{v}_{i}", cat="Binary")
                prob += delta[v, i] == pulp.lpSum(x[k, i, v] for k in N if k != i), f"Delta_{v}_{i}"
        
        # 10) Akış Korunumu
        for v in vehicle_ids:
            for k in M_route:
                prob += (pulp.lpSum(x[i, k, v] for i in N if i != k) ==
                         pulp.lpSum(x[k, j, v] for j in N if j != k)), f"Flow_{k}_{v}"
        
        # 11) Ziyaret Kısıtları
        # a) Home delivery yapılacak müşteri => rota üzerinde tam 1 kez ziyaret
        for i in M:
            prob += pulp.lpSum(delta[v, i] for v in vehicle_ids) == home[i], f"CustomerVisit_{i}"
        # b) Locker kullanılacaksa (w[j]=1), rota üzerinde tam 1 kez ziyaret
        for j in L:
            prob += pulp.lpSum(x[i, j, v] for i in N if i != j for v in vehicle_ids) == w[j], f"Visit_Locker_{j}"
        
        # 12) Depo çıkış ve dönüş
        for v in vehicle_ids:
            prob += pulp.lpSum(x[d, j, v] for d in D for j in N if d != j) == y[v], f"Depart_{v}"
            prob += pulp.lpSum(x[i, d, v] for d in D for i in N if d != i) == y[v], f"Return_{v}"
        
        # 13) Zaman (Time) Kısıtı
        speed = 60.0
        for v in vehicle_ids:
            prob += (pulp.lpSum((self.dist_matrix[i][j] / speed) * x[i, j, v]
                       for i in N for j in N if i != j)
                     <= vehicles[v]['max_duration']), f"TimeLimit_{v}"
        
        # 14) Forbidden Group Kısıtları
        for group in forbidden_groups:
            for v in vehicle_ids:
                prob += pulp.lpSum(x[i, j, v] for i in group for j in group if i != j) <= 1, f"Forbidden_{v}_{str(group)}"
        
        # 15) Amaç Fonksiyonu
        routing_cost = pulp.lpSum(
            vehicles[v]['fixed_cost'] * y[v] +
            pulp.lpSum(self.dist_matrix[i][j] * vehicles[v]['cost_per_km'] * x[i, j, v]
                       for i in N for j in N if i != j)
            for v in vehicle_ids
        )
        assignment_cost = pulp.lpSum(
            self.dist_matrix[i][j] * self.data.loc[i, 'cost'] * z[i, j]
            for i in M for j in candidate_lockers[i] if (i, j) in z
        )
        prob += routing_cost + assignment_cost, "Total_Cost"
        
        # 16) Modelin Çözümü
        if time_limit:
            solver_instance = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=True)
            prob.solve(solver_instance)
        else:
            prob.solve()
        
        print("Solver Status:", pulp.LpStatus[prob.status])
        
        # 17) Çözüm Sonuçlarının Çıkarılması
        # a) Rotalar
        routes = {}
        for v in vehicle_ids:
            if pulp.value(y[v]) < 0.5:
                continue
            # Depodan çıkan ilk ok
            start_depot = None
            for dpt in D:
                for j in N:
                    if dpt != j and pulp.value(x[dpt, j, v]) > 0.5:
                        start_depot = dpt
                        break
                if start_depot is not None:
                    break
            
            if start_depot is None:
                # Eğer hiç çıkış yoksa, rota basitçe depot->depot
                routes[v] = [D[0], D[0]]
                continue
            
            # Rota reconstruct
            route_path = [start_depot]
            visited_set = {start_depot}
            current = start_depot
            while True:
                next_node = None
                for j in N:
                    if j != current:
                        val = pulp.value(x[current, j, v])
                        if val is not None and val > 0.5:
                            next_node = j
                            break
                if next_node is None or next_node in visited_set:
                    if route_path[-1] not in D:
                        route_path.append(D[0])
                    break
                else:
                    route_path.append(next_node)
                    visited_set.add(next_node)
                    current = next_node
            
            routes[v] = route_path
        
        # b) Müşteri–Locker Eşleşmeleri
        locker_assignments = {}
        for i in M:
            # Check if home[i] = 1 => no locker
            if pulp.value(home[i]) > 0.5:
                # Ev teslimatı
                locker_assignments[i] = None
            else:
                # locker arayalım
                for j in candidate_lockers[i]:
                    if (i, j) in z and pulp.value(z[i, j]) > 0.5:
                        locker_assignments[i] = j
                        break
                else:
                    # hiçi bulamazsa None
                    locker_assignments[i] = None
    
        total_cost = pulp.value(prob.objective) if prob.status == 1 else None
        
        # 3. dönen değer: locker_assignments
        return routes, total_cost, locker_assignments
    

    ##########################################################
    # Heuristic Yöntem – Feasible Sweep Algoritması
    ##########################################################
    def solve_vrp_heuristic(self, vehicles_df, forbidden_groups=None):
        if forbidden_groups is None:
            forbidden_groups = []
        speed = 60.0
        max_capacity = vehicles_df['capacity'].max()
        max_duration_minutes = vehicles_df['max_duration'].max()
        max_distance = (max_duration_minutes / 60.0) * speed
        N = list(range(self.data.shape[0]))
        D = self.depot_indices

        # (A) Her non‑depot node, en yakın depoya göre gruplandırılsın.
        depot_groups = {d: [] for d in D}
        for i in N:
            if i in D:
                continue
            d_nearest = min(D, key=lambda d: self.dist_matrix[d][i])
            depot_groups[d_nearest].append(i)

        # (B) Her depo grubu için sweep algoritması.
        routes_all = []
        for depot, nodes in depot_groups.items():
            if not nodes:
                continue
            depot_coord = self.coordinates[depot]
            sorted_nodes = sorted(nodes, key=lambda i: math.atan2(
                self.data.loc[i, 'Latitude'] - depot_coord[0],
                self.data.loc[i, 'Longitude'] - depot_coord[1]
            ))
            current_route = [depot]
            current_demand = 0.0
            current_distance = 0.0
            for node in sorted_nodes:
                node_demand = self.demands[node]
                candidate_id = self.data.loc[node, 'ID']
                route_ids = [self.data.loc[n, 'ID'] for n in current_route if n != depot]
                conflict = any(candidate_id in group and any(rid in group for rid in route_ids)
                               for group in forbidden_groups)
                if conflict:
                    current_route.append(depot)
                    routes_all.append(current_route)
                    current_route = [depot]
                    current_demand = 0.0
                    current_distance = 0.0
                last_node = current_route[-1]
                extra = self.dist_matrix[last_node][node] + self.dist_matrix[node][depot] - self.dist_matrix[last_node][depot]
                if (current_demand + node_demand <= max_capacity) and (current_distance + extra <= max_distance):
                    current_route.append(node)
                    current_demand += node_demand
                    current_distance += extra
                else:
                    current_route.append(depot)
                    routes_all.append(current_route)
                    current_route = [depot, node]
                    current_demand = node_demand
                    current_distance = self.dist_matrix[depot][node] + self.dist_matrix[node][depot]
            if current_route[-1] != depot:
                current_route.append(depot)
            routes_all.append(current_route)

        # (C) Yerel iyileştirme ve forbidden kontrolü.
        optimized_routes = []
        for route in routes_all:
            if len(route) > 3:
                route_opt = two_opt(route, self.dist_matrix)
                route_opt = three_opt(route_opt, self.dist_matrix)
            else:
                route_opt = route
            route_opt = self.fix_route_forbidden(route_opt, forbidden_groups)
            optimized_routes.append(route_opt)

        # (D) Araçlara atama
        vehicles_sorted = vehicles_df.sort_values(by='capacity', ascending=False)
        route_assignments = {}
        i = 0
        for idx, row in vehicles_sorted.iterrows():
            if i < len(optimized_routes):
                route = optimized_routes[i]
                route_demand = sum(self.demands[node] for node in route if node not in D)
                route_distance = compute_route_distance(route, self.dist_matrix)
                route_assignments[row['vehicle_id']] = {
                    'route': route,
                    'distance': route_distance,
                    'demand': route_demand,
                    'vehicle': row.to_dict()
                }
                i += 1
        if i < len(optimized_routes):
            highest_vehicle = vehicles_sorted.iloc[0]
            existing = route_assignments.get(highest_vehicle['vehicle_id'], {})
            if existing and 'routes' in existing:
                routes_list = existing['routes']
            else:
                routes_list = [existing['route']] if existing else []
            for r in optimized_routes[i:]:
                routes_list.append(r)
            total_distance = sum(compute_route_distance(r, self.dist_matrix) for r in routes_list)
            total_demand = sum(sum(self.demands[node] for node in r if node not in D) for r in routes_list)
            route_assignments[highest_vehicle['vehicle_id']] = {
                'route': routes_list,
                'distance': total_distance,
                'demand': total_demand,
                'vehicle': highest_vehicle.to_dict()
            }
        return route_assignments, []

    def solve_vrp_heuristic_with_sa(self, vehicles_df, forbidden_groups=None):
        """
        Heuristic çözüm üzerinde Simulated Annealing (SA) uygulayarak rota iyileştirmesi yapar.
        - İlk çözüm, solve_vrp_heuristic ile oluşturulur.
        - Her rota üzerinde SA uygulanarak, rota sıralaması (ve dolayısıyla mesafe ve maliyet) iyileştirilir.
        - Son olarak, her rota için araç ataması maliyetler üzerinden tekrar değerlendirilir.
        - Bu adım, araçların km başına maliyetleri ve sabit maliyetleri göz önünde bulundurarak en uygun atanmayı sağlar.
        """
        # Adım 1: Temel heuristic çözüm
        route_assignments, _ = self.solve_vrp_heuristic(vehicles_df, forbidden_groups)
    
        # Adım 2: Her rota üzerinde SA uygulayarak iyileştirme
        sa_optimized_routes = {}
        for vid, rdata in route_assignments.items():
            initial_route = rdata['route']
            best_route, best_distance = simulated_annealing(initial_route, self.dist_matrix)
            sa_optimized_routes[vid] = {
                'route': best_route,
                'distance': best_distance,
                'demand': rdata['demand'],
                'vehicle': rdata['vehicle']
            }
    
        # Adım 3: Araç Maliyetlerini Göz Önüne Alarak Rota Ataması
        final_route_assignments = {}
        for vid, rdata in sa_optimized_routes.items():
            route = rdata['route']
            route_distance = compute_route_distance(route, self.dist_matrix)
            best_vehicle = None
            best_vehicle_cost = float('inf')
            for v in vehicles_df['vehicle_id']:
                v_row = vehicles_df[vehicles_df['vehicle_id'] == v].iloc[0]
                cost = v_row['fixed_cost'] + route_distance * v_row['cost_per_km']
                if cost < best_vehicle_cost:
                    best_vehicle_cost = cost
                    best_vehicle = v
            final_route_assignments[best_vehicle] = {
                'route': route,
                'distance': route_distance,
                'demand': rdata['demand'],
                'vehicle': vehicles_df[vehicles_df['vehicle_id'] == best_vehicle].iloc[0].to_dict()
            }
        return final_route_assignments, []

    ##########################################################
    # OSRM tabanlı interaktif rota görselleştirme
    ##########################################################
    def create_advanced_route_map(self, route_costs, data_source, locker_assignments=None):
        """
        route_costs: {
            vehicle_id: {
                'route': [...],
                'distance': float,
                'cost': float,
                ... 
            }, 
            ...
        }
        data_source: Gösterimde kullanılacak DataFrame (örneğin st.session_state.data_for_vrp)
        locker_assignments: { customer_i: locker_j or None } 
            => Hangi müşteri hangi lockera atandı (eğer None ise ev teslimatı)
        """
        center_lat = self.data['Latitude'].mean()
        center_lon = self.data['Longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        colors = [
            'red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue',
            'cadetblue', 'darkgreen', 'darkpurple', 'pink', 'lightred', 'beige',
            'lightblue', 'lightgreen', 'gray'
        ]
        
        # 1) Nokta Marker'ları
        for idx, row in data_source.iterrows():
            node_type = row['node_type']
            lat, lon = row['Latitude'], row['Longitude']
    
            if node_type == 'depot':
                marker_color = 'black'
                icon_ = 'factory'
                popup_text = f"Depo (ID: {row['ID']})"
            elif node_type == 'locker':
                marker_color = 'gray'
                icon_ = 'lock'
                popup_text = f"Locker (ID: {row['ID']}) - Kapasite: {row.get('remaining_capacity', row['demand'])}"
            else:
                marker_color = 'blue'
                icon_ = 'home'
                popup_text = f"Müşteri (ID: {row['ID']}) - Talep: {row['demand']}"
    
            folium.Marker(
                [lat, lon],
                popup=popup_text,
                icon=folium.Icon(color=marker_color, icon=icon_)
            ).add_to(m)
        
        # 2) Rota Segmentlerinin Çizimi
        osrm_cache = {}
        for idx, (vid, rdata) in enumerate(route_costs.items()):
            route_val = rdata.get('route', None)
            if route_val is None or not isinstance(route_val, list) or len(route_val) < 2:
                continue
            
            # Tek liste ise liste içinde liste yap
            if not isinstance(route_val[0], list):
                routes = [route_val]
            else:
                routes = route_val
            
            color = colors[idx % len(colors)]
            total_distance = rdata.get('distance', 0)
            total_cost = rdata.get('cost', 0)
            
            for sub_route in routes:
                if not isinstance(sub_route, list) or len(sub_route) < 2:
                    continue
                
                full_route_coords = []
                for i in range(len(sub_route)-1):
                    start_node = sub_route[i]
                    end_node = sub_route[i+1]
                    start_coord = (self.data.iloc[start_node]['Latitude'], self.data.iloc[start_node]['Longitude'])
                    end_coord = (self.data.iloc[end_node]['Latitude'], self.data.iloc[end_node]['Longitude'])
                    segment_coords = self.get_osrm_route(start_coord, end_coord, osrm_cache)
                    if not segment_coords:
                        segment_coords = [start_coord, end_coord]
                    if full_route_coords:
                        full_route_coords.extend(segment_coords[1:])
                    else:
                        full_route_coords.extend(segment_coords)
                
                popup_text = f"Araç {vid} - Toplam Mesafe: {total_distance:.2f} km"
                if total_cost:
                    popup_text += f", Maliyet: {total_cost:.2f}"
                
                folium.PolyLine(
                    full_route_coords,
                    color=color,
                    weight=4,
                    opacity=0.7,
                    popup=popup_text,
                    tooltip=f"Araç ID: {vid}"
                ).add_to(m)
        
        # 3) Müşteri–Locker Eşleşmeleri İçin Dotted Line
        if locker_assignments is not None:
            for cust_i, locker_j in locker_assignments.items():
                if locker_j is None:
                    # Müşteri evine teslim alıyor
                    continue
                # Customer i ile locker j arasında kesikli (dotted) çizgi
                cust_lat = self.data.iloc[cust_i]['Latitude']
                cust_lon = self.data.iloc[cust_i]['Longitude']
                lock_lat = self.data.iloc[locker_j]['Latitude']
                lock_lon = self.data.iloc[locker_j]['Longitude']
                
                folium.PolyLine(
                    locations=[(cust_lat, cust_lon), (lock_lat, lock_lon)],
                    color='black',  # Örn. siyah
                    weight=2,       # İncelik
                    dash_array='5,5',  # Kesikli çizgi
                    popup=f"Müşteri {cust_i} -> Locker {locker_j}",
                    tooltip="Locker ataması"
                ).add_to(m)
        
        folium.LayerControl().add_to(m)
        return m

###############################################
# Streamlit Uygulaması – Dosya Yükleme ve Node Yönetimi
###############################################
st.set_page_config(page_title="Gelişmiş VRP Çözümleme", layout="wide")

# Tema stilini uygula (Koyu arka plan + açık renk yazı)
dark_theme = """
<style>
    /* Global ayarlar */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #121212;
        color: #F5F5F5;
        font-family: 'Roboto', sans-serif;
    }
    .stApp {
        background-color: #121212;
        color: #F5F5F5;
    }
    /* Sidebar ayarları */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio,
    [data-testid="stSidebar"] .stCheckbox,
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] .stCaption {
        color: #FFFFFF !important;
    }
    /* Butonlar */
    .stButton > button {
        background-color: #333;
        color: #F5F5F5;
        border: none;
    }
    .stButton > button:hover {
        background-color: #444;
        color: #FFFFFF;
    }
    /* Metin kutuları, file uploader vb. */
    .stTextInput > div > div > input {
        background-color: #333;
        color: #F5F5F5;
    }
    .stFileUploader > div, .stFileUploader label, .stFileUploader span {
        background-color: #333;
        color: #F5F5F5 !important;
    }
    .stDownloadButton > button {
        background-color: #333;
        color: #F5F5F5;
    }
</style>
"""
st.markdown(dark_theme, unsafe_allow_html=True)

st.title("🚚 :blue[Gelişmiş Araç Rotalama Problemi Çözücü (İstanbul)]")

# ---------------------------------------------------------------------
# Sidebar’da forbidden node grupları
# ---------------------------------------------------------------------
st.sidebar.subheader(":blue[Aynı Rotada Bulunamayacak Nodelar]")
if 'forbidden_groups' not in st.session_state:
    st.session_state.forbidden_groups = []

all_node_ids = []
if st.session_state.get('original_data') is not None:
    all_node_ids = sorted(st.session_state.original_data['ID'].unique())
else:
    all_node_ids = list(range(100))

selected_nodes = st.sidebar.multiselect(
    ":blue[Rota içinde yan yana bulunmasın istenen node ID’leri:]",
    options=all_node_ids,
    key='forbidden_nodes_multiselect'
)

if st.sidebar.button(":blue[Ekle]"):
    if selected_nodes:
        st.session_state.forbidden_groups.append(selected_nodes)
        st.sidebar.success(f"Eklendi: {selected_nodes}")
    else:
        st.sidebar.warning(":blue[Lütfen en az bir node seçiniz.]")

if st.session_state.forbidden_groups:
    st.sidebar.write(":blue[Eklenen forbidden gruplar:]")
    for grp in st.session_state.forbidden_groups:
        st.sidebar.write(grp)

# ---------------------------------------------------------------------
# Veri Yükleme Paneli
# ---------------------------------------------------------------------
st.title(":blue[Veri Yükleme Paneli] 📂")

st.markdown("""
**Lütfen aşağıdaki dosyaları yükleyin:**
- **Lokasyon Dosyası**: `Latitude`, `Longitude`, `demand`, `node_type` kolonlarını içermelidir.
- **Araç Dosyası**: `vehicle_id`, `capacity`, `max_duration`, `cost_per_km`, `fixed_cost` kolonlarını içermelidir.
""")

import io

def create_example_file():
    sample_data = {
        "ID": ["Depo1", "Customer2", "Locker3"],
        "Latitude": [41.0082, 41.0151, 41.0123],
        "Longitude": [28.9784, 28.9744, 28.9800],
        "demand": [0, 5, 10],
        "node_type": ["depot", "customer", "locker"],
        "cost": [0, 1.0, 0]  # Cost bilgisi; müşteri için locker atama maliyeti
    }
    df = pd.DataFrame(sample_data)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output

# Örnek dosya indirme
st.download_button(
    label="📥 :blue[Örnek Lokasyon Dosyasını İndir]",
    data=create_example_file(),
    file_name="ornek_lokasyon.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Dosya yükleme
st.markdown("### 📍 :blue[Lokasyon Dosyası Yükle]")
uploaded_nodes_file = st.file_uploader(":blue[Lokasyon Dosyası (Excel formatında)]", type="xlsx", key="nodes")

st.markdown("### 🚗 :blue[Araç Dosyası Yükle]")
uploaded_vehicles_file = st.file_uploader(":blue[Araç Dosyası (Excel formatında)]", type="xlsx", key="vehicles")

def preview_uploaded_file(file, file_type):
    if file is not None:
        try:
            df = pd.read_excel(file)
            st.write(f"📋 **{file_type} Dosyası Önizleme:**")
            st.dataframe(df.head())
            return df
        except Exception as e:
            st.error(f"❌ Hata: Dosya okunamadı! ({str(e)})")
            return None
    return None

# Yüklenen dosyaların önizlemesini göster
original_data = preview_uploaded_file(uploaded_nodes_file, "Lokasyon")
vehicles_df = preview_uploaded_file(uploaded_vehicles_file, "Araç")

# ---------------------------------------------------------------------
# Eğer iki dosya da yüklenmişse, verileri işle
# ---------------------------------------------------------------------
if uploaded_nodes_file is not None and uploaded_vehicles_file is not None:
    with st.spinner(":blue[Dosyalar yükleniyor...] ⏳"):
        time.sleep(1)

    st.success("✔️ :blue[Dosyalar başarıyla yüklendi!]")

    # Progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i + 1)

    # DataFrame
    original_data = pd.read_excel(uploaded_nodes_file)
    vehicles_df = pd.read_excel(uploaded_vehicles_file)

    st.session_state.original_data = original_data.copy()

    needed_node_cols = ['Latitude', 'Longitude', 'demand', 'node_type']
    needed_vehicle_cols = ['vehicle_id', 'capacity', 'max_duration', 'cost_per_km', 'fixed_cost']

    if not all(col in original_data.columns for col in needed_node_cols):
        st.error(f"❌ Lokasyon dosyasında şu kolonlar eksik veya hatalı: {needed_node_cols}")
        st.stop()

    if not all(col in vehicles_df.columns for col in needed_vehicle_cols):
        st.error(f"❌ Araç dosyasında şu kolonlar eksik veya hatalı: {needed_vehicle_cols}")
        st.stop()

    # Depo, locker, müşteri ayrıştırma
    depot_df = original_data[original_data['node_type'] == 'depot'].copy()
    locker_df = original_data[original_data['node_type'] == 'locker'].copy()
    customer_df = original_data[original_data['node_type'] == 'customer'].copy()

    if depot_df.empty:
        st.error("❌ :blue[En az bir depo bulunmalıdır!]")
        st.stop()

    # Locker'lar için remaining_capacity
    if not locker_df.empty:
        locker_df['remaining_capacity'] = locker_df['demand']

    max_lock_distance = st.sidebar.number_input(
        ":blue[Maksimum Locker Atama Mesafesi (km):]", 
        min_value=0.1, 
        value=2.0, 
        step=0.1
    )

    # Verileri bir araya getir
    data_for_vrp = pd.concat([depot_df, locker_df, customer_df], ignore_index=True)

    st.session_state.data_for_vrp = data_for_vrp.copy()
    if "vehicles_df" not in st.session_state:
        st.session_state.vehicles_df = vehicles_df.copy()

    st.success("✅ :blue[Veriler başarıyla işlendi ve kaydedildi!]")

    # ---------------------------------------------------------
    # Node Yönetimi (Sil / Ekle)
    # ---------------------------------------------------------
    node_management_mode = st.sidebar.radio("Node Yönetimi", ["Yok", "Sil", "Ekle"])

    if node_management_mode == "Sil":
        st.sidebar.info(":blue[Haritada tıklayarak silmek istediğiniz customer veya locker’ı seçin.]")
        solver = AdvancedVRPSolver(st.session_state.data_for_vrp)
        # Heuristic rota çiz, silinecek node'u bul
        route_costs, _ = solver.solve_vrp_heuristic(
            st.session_state.vehicles_df, 
            forbidden_groups=st.session_state.forbidden_groups
        )
        current_map = solver.create_advanced_route_map(route_costs, st.session_state.data_for_vrp)
        map_data = st_folium(current_map, width=700, height=500)
        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]
            df = st.session_state.data_for_vrp
            df_non_depot = df[df["node_type"] != "depot"]
            if not df_non_depot.empty:
                distances = df_non_depot.apply(
                    lambda row: distance.distance(
                        (row["Latitude"], row["Longitude"]),
                        (click_lat, click_lon)
                    ).kilometers,
                    axis=1
                )
                closest_idx = distances.idxmin()
                closest_distance = distances.min()
                if closest_distance < 0.5:
                    st.write("Silinecek Node Bilgisi:", df_non_depot.loc[closest_idx])
                    if st.button("Bu node'u sil"):
                        st.session_state.data_for_vrp = st.session_state.data_for_vrp.drop(closest_idx).reset_index(drop=True)
                        st.success("Node silindi. Lütfen 'Rota Yenile' butonuna basarak rotayı güncelleyin.")

    elif node_management_mode == "Ekle":
        st.sidebar.info(":blue[Yeni node eklemek için bilgileri girin.]")
        with st.form(key="add_node_form"):
            lat = st.number_input("Enlem", value=st.session_state.data_for_vrp["Latitude"].mean())
            lon = st.number_input("Boylam", value=st.session_state.data_for_vrp["Longitude"].mean())
            demand = st.number_input("Demand", value=1.0)
            node_type = st.selectbox("Node Tipi", ["customer", "locker"])
            submit = st.form_submit_button("Ekle")
            if submit:
                if st.session_state.data_for_vrp.empty:
                    new_id = 0
                else:
                    if "ID" in st.session_state.data_for_vrp.columns:
                        new_id = st.session_state.data_for_vrp["ID"].max() + 1
                    else:
                        new_id = len(st.session_state.data_for_vrp)
                
                new_row = {
                    "ID": new_id,
                    "Latitude": lat,
                    "Longitude": lon,
                    "demand": demand,
                    "node_type": node_type
                }
                if node_type == "locker":
                    new_row["remaining_capacity"] = demand
                
                st.session_state.data_for_vrp = st.session_state.data_for_vrp.append(new_row, ignore_index=True)
                st.success("Yeni node eklendi. Lütfen 'Rota Yenile' butonuna basarak rotayı güncelleyin.")

    # ---------------------------------------------------------
    # Rota Çözüm Yöntemi Seçimi
    # ---------------------------------------------------------
    method = st.sidebar.selectbox("Çözüm Yöntemi Seçiniz:", [
        "Heuristic (Hızlı)",
        "MILP (Optimal Ama Yavaş)",
        "Heuristic + Simulated Annealing (SA)"
    ])

    # ---------------------------------------------------------
    # Heuristic + SA
    # ---------------------------------------------------------
    if method == "Heuristic + Simulated Annealing (SA)":
        solver = AdvancedVRPSolver(st.session_state.data_for_vrp)
        route_costs, _ = solver.solve_vrp_heuristic_with_sa(
            st.session_state.vehicles_df,
            forbidden_groups=st.session_state.forbidden_groups
        )
        total_cost = sum(
            rdata['vehicle']['fixed_cost'] + rdata['vehicle']['cost_per_km'] * rdata['distance']
            for rdata in route_costs.values()
        )

        st.header("Rotalama Sonuçları (Heuristic + SA)")
        st.metric("Toplam Maliyet", f"{total_cost:.2f} TL")
        with st.expander("Rota Detayları"):
            for vid, rdata in route_costs.items():
                route_ids = [str(solver.data.iloc[node]['ID']) for node in rdata['route']]
                st.write(
                    f"Araç {vid}: Rota (ID'ler) -> {route_ids} | "
                    f"Mesafe: {rdata['distance']:.2f} km | Toplam Talep: {rdata['demand']:.2f}"
                )
        route_map = solver.create_advanced_route_map(route_costs, st.session_state.data_for_vrp)
        folium_static(route_map, width=1000, height=600)

    # ---------------------------------------------------------
    # MILP (Optimal)
    # ---------------------------------------------------------
    elif method == "MILP (Optimal Ama Yavaş)":
        solver = AdvancedVRPSolver(st.session_state.data_for_vrp)
        # Yeni solve_vrp_milp => 3 sonuç döndürüyor: routes, cost, locker_assignments
        routes, total_cost, locker_assignments = solver.solve_vrp_milp(
            st.session_state.vehicles_df,
            time_limit=st.sidebar.number_input("MILP Zaman Limiti (saniye):", min_value=1, value=600, step=1),
            locker_km_limit=max_lock_distance
        )
        
        if routes is not None and total_cost is not None:
            st.header("Rotalama Sonuçları (MILP)")
            st.metric("Toplam Maliyet", f"{total_cost:.2f} TL")
            with st.expander("Rota Detayları"):
                for vid, route in routes.items():
                    if route:
                        route_ids = [str(solver.data.iloc[node]['ID']) for node in route]
                        st.write(f"Araç {vid}: Ziyaret Sırası (ID'ler) -> {route_ids}")

            # Rota görselleştirmesi için route_costs sözlüğü
            # Burada distance hesaplanır, cost None bırakılabilir veya siz de hesap yapabilirsiniz
            route_costs = {}
            for vid, route in routes.items():
                dist = compute_route_distance(route, solver.dist_matrix)
                route_costs[vid] = {
                    'route': route,
                    'distance': dist,
                    'cost': None,  # isterseniz sabit+km maliyetini hesaplayabilirsiniz
                    'vehicle': {}
                }
            
            # Haritada dotted line için locker_assignments da ekleniyor
            route_map = solver.create_advanced_route_map(
                route_costs=route_costs,
                data_source=st.session_state.data_for_vrp,
                locker_assignments=locker_assignments  # Müşteri–locker eşleşmeleri
            )
            folium_static(route_map, width=1000, height=600)

            # Locker atamaları tablosu
            with st.expander("Locker Atamaları"):
                # locker_assignments = { customer_index: locker_index or None }
                assigned_locker_info = []
                for cust_i, lock_j in locker_assignments.items():
                    if lock_j is None:
                        assigned_locker_info.append({
                            "CustomerID": str(solver.data.iloc[cust_i]['ID']),
                            "LockerID": "Ev Teslimatı (None)"
                        })
                    else:
                        assigned_locker_info.append({
                            "CustomerID": str(solver.data.iloc[cust_i]['ID']),
                            "LockerID": str(solver.data.iloc[lock_j]['ID'])
                        })
                st.dataframe(pd.DataFrame(assigned_locker_info))
        else:
            st.error("MILP çözümü bulunamadı veya uygun değil.")

    # ---------------------------------------------------------
    # Heuristic (Hızlı) 
    # ---------------------------------------------------------
    else:
        with st.spinner('Rota hesaplanıyor...'):
            solver = AdvancedVRPSolver(st.session_state.data_for_vrp)
            route_costs, _ = solver.solve_vrp_heuristic(
                st.session_state.vehicles_df,
                forbidden_groups=st.session_state.forbidden_groups
            )
            total_cost = sum(
                rdata['vehicle']['fixed_cost'] + rdata['vehicle']['cost_per_km'] * rdata['distance']
                for rdata in route_costs.values()
            )

            st.header("Rotalama Sonuçları (Heuristic)")
            st.metric("Toplam Maliyet", f"{total_cost:.2f} TL")
            with st.expander("Rota Detayları"):
                for vid, rdata in route_costs.items():
                    if isinstance(rdata['route'], list) and len(rdata['route']) > 0 and isinstance(rdata['route'][0], list):
                        route_ids = [
                            [str(solver.data.iloc[node]['ID']) for node in r]
                            for r in rdata['route']
                        ]
                    elif isinstance(rdata['route'], list):
                        route_ids = [str(solver.data.iloc[node]['ID']) for node in rdata['route']]
                    else:
                        route_ids = []
                    st.write(
                        f"Araç {vid}: Rota (ID'ler) -> {route_ids} | "
                        f"Mesafe: {rdata['distance']:.2f} km | Toplam Talep: {rdata['demand']:.2f}"
                    )
            route_map = solver.create_advanced_route_map(route_costs, st.session_state.data_for_vrp)
            folium_static(route_map, width=1000, height=600)

else:
    st.warning("Lütfen hem lokasyon/tip bilgilerini hem de araç bilgilerini içeren Excel dosyalarını yükleyiniz.")

# ---------------------------------------------------------
# Rota Yenile Butonu (Heuristic hızla tekrar)
# ---------------------------------------------------------
if st.button("Rota Yenile"):
    solver = AdvancedVRPSolver(st.session_state.data_for_vrp)
    route_costs, _ = solver.solve_vrp_heuristic(
        st.session_state.vehicles_df,
        forbidden_groups=st.session_state.forbidden_groups
    )
    st.session_state.last_route_costs = route_costs
    updated_map = solver.create_advanced_route_map(route_costs, st.session_state.data_for_vrp)
    folium_static(updated_map, width=1000, height=600)
    st.success("Rota güncellendi!")
