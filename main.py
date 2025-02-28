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
import utils
from utils import compute_route_distance, two_opt, three_opt, simulated_annealing
from vrp_solver import AdvancedVRPSolver


###############################################
# Streamlit Uygulaması – Dosya Yükleme ve Node Yönetimi
###############################################

st.set_page_config(page_title="Gelişmiş VRP Çözümleme", layout="wide")
st.title("🚚 Gelişmiş Araç Rotalama Problemi Çözücü (İstanbul)")

# Sidebar’da forbidden node grupları (önceki kısım)…
st.sidebar.subheader("Aynı Rotada Bulunamayacak Nodelar")
if 'forbidden_groups' not in st.session_state:
    st.session_state.forbidden_groups = []
all_node_ids = []
if st.session_state.get('original_data') is not None:
    all_node_ids = sorted(st.session_state.original_data['ID'].unique())
else:
    all_node_ids = list(range(100))
selected_nodes = st.sidebar.multiselect("Rota içinde yan yana bulunmasın istenen node ID’leri:", options=all_node_ids, key='forbidden_nodes_multiselect')
if st.sidebar.button("Ekle"):
    if selected_nodes:
        st.session_state.forbidden_groups.append(selected_nodes)
        st.sidebar.success(f"Eklendi: {selected_nodes}")
    else:
        st.sidebar.warning("Lütfen en az bir node seçiniz.")
if st.session_state.forbidden_groups:
    st.sidebar.write("Eklenen forbidden gruplar:")
    for grp in st.session_state.forbidden_groups:
        st.sidebar.write(grp)

# Dosya yükleme
uploaded_nodes_file = st.file_uploader("Lokasyon Dosyası Yükle", type="xlsx", key="nodes")
uploaded_vehicles_file = st.file_uploader("Araç Dosyası Yükle", type="xlsx", key="vehicles")

if uploaded_nodes_file is not None and uploaded_vehicles_file is not None:
    original_data = pd.read_excel(uploaded_nodes_file)
    vehicles_df = pd.read_excel(uploaded_vehicles_file)
    st.session_state.original_data = original_data.copy()
    needed_node_cols = ['Latitude', 'Longitude', 'demand', 'node_type', 'deliver_type']
    if not all(col in original_data.columns for col in needed_node_cols):
        st.error(f"Lokasyon dosyasında şu kolonlar eksik veya hatalı: {needed_node_cols}")
        st.stop()
    needed_vehicle_cols = ['vehicle_id', 'capacity', 'max_duration', 'cost_per_km', 'fixed_cost']
    if not all(col in vehicles_df.columns for col in needed_vehicle_cols):
        st.error(f"Araç dosyasında şu kolonlar eksik veya hatalı: {needed_vehicle_cols}")
        st.stop()
    depot_df = original_data[original_data['node_type'] == 'depot'].copy()
    locker_df = original_data[original_data['node_type'] == 'locker'].copy()
    customer_df = original_data[original_data['node_type'] == 'customer'].copy()
    if depot_df.empty:
        st.error("En az bir depo bulunmalıdır!")
        st.stop()
    if not locker_df.empty:
        locker_df['remaining_capacity'] = locker_df['demand']
    max_lock_distance = st.sidebar.number_input("Maksimum Locker Atama Mesafesi (km):", min_value=0.1, value=2.0, step=0.1)
    last_mile_mask = (customer_df['deliver_type'] == 'last_mile')
    for idx, row in customer_df[last_mile_mask].iterrows():
        cust_coord = (row['Latitude'], row['Longitude'])
        candidate_lockers = []
        for l_idx, l_row in locker_df.iterrows():
            locker_coord = (l_row['Latitude'], l_row['Longitude'])
            d_km = distance.distance(cust_coord, locker_coord).kilometers
            if d_km <= max_lock_distance and l_row['remaining_capacity'] >= row['demand']:
                candidate_lockers.append((l_idx, d_km))
        if candidate_lockers:
            candidate_lockers.sort(key=lambda x: x[1])
            chosen_locker_idx = candidate_lockers[0][0]
            locker_df.at[chosen_locker_idx, 'remaining_capacity'] -= row['demand']
            assigned_locker_id = locker_df.at[chosen_locker_idx, 'ID'] if 'ID' in locker_df.columns else chosen_locker_idx
            customer_df.at[idx, 'assigned_locker'] = assigned_locker_id
            customer_df.drop(idx, inplace=True)
        else:
            customer_df.at[idx, 'deliver_type'] = 'last_feet'
    data_for_vrp = pd.concat([depot_df, locker_df, customer_df], ignore_index=True)
    # Veriyi session_state’de saklayalım.
    st.session_state.data_for_vrp = data_for_vrp.copy()
    if "vehicles_df" not in st.session_state:
        st.session_state.vehicles_df = vehicles_df.copy()

    # ---------------------------
    # Node Yönetimi (Sil / Ekle) Kontrolleri
    # ---------------------------
    node_management_mode = st.sidebar.radio("Node Yönetimi", ["Yok", "Sil", "Ekle"])
    if node_management_mode == "Sil":
        st.sidebar.info("Haritada tıklayarak silmek istediğiniz customer veya locker’ı seçin.")
        # Solver'ı güncel veriden oluşturuyoruz.
        solver = AdvancedVRPSolver(st.session_state.data_for_vrp)
        route_costs, _ = solver.solve_vrp_heuristic(st.session_state.vehicles_df, forbidden_groups=st.session_state.forbidden_groups)
        # Haritayı oluşturuyoruz (veri olarak güncel data_for_vrp kullanıyoruz)
        current_map = solver.create_advanced_route_map(route_costs, st.session_state.data_for_vrp)
        map_data = st_folium(current_map, width=700, height=500)
        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]
            df = st.session_state.data_for_vrp
            # Sadece customer ve locker'ları hedefleyelim.
            df_non_depot = df[df["node_type"] != "depot"]
            if not df_non_depot.empty:
                distances = df_non_depot.apply(lambda row: distance.distance((row["Latitude"], row["Longitude"]), (click_lat, click_lon)).kilometers, axis=1)
                closest_idx = distances.idxmin()
                closest_distance = distances.min()
                if closest_distance < 0.5:  # 500 m içinde ise
                    st.write("Silinecek Node Bilgisi:", df_non_depot.loc[closest_idx])
                    if st.button("Bu node'u sil"):
                        st.session_state.data_for_vrp = st.session_state.data_for_vrp.drop(closest_idx).reset_index(drop=True)
                        st.success("Node silindi. Lütfen 'Rota Yenile' butonuna basarak rotayı güncelleyin.")
    elif node_management_mode == "Ekle":
        st.sidebar.info("Yeni node eklemek için bilgileri girin.")
        with st.form(key="add_node_form"):
            lat = st.number_input("Enlem", value=st.session_state.data_for_vrp["Latitude"].mean())
            lon = st.number_input("Boylam", value=st.session_state.data_for_vrp["Longitude"].mean())
            demand = st.number_input("Demand", value=1.0)
            node_type = st.selectbox("Node Tipi", ["customer", "locker"])
            deliver_type = "last_feet" if node_type == "customer" else "none"
            submit = st.form_submit_button("Ekle")
            if submit:
                if st.session_state.data_for_vrp.empty:
                    new_id = 0
                else:
                    new_id = st.session_state.data_for_vrp["ID"].max() + 1
                new_row = {"ID": new_id, "Latitude": lat, "Longitude": lon, "demand": demand,
                           "node_type": node_type, "deliver_type": deliver_type}
                if node_type == "locker":
                    new_row["remaining_capacity"] = demand
                st.session_state.data_for_vrp = st.session_state.data_for_vrp.append(new_row, ignore_index=True)
                st.success("Yeni node eklendi. Lütfen 'Rota Yenile' butonuna basarak rotayı güncelleyin.")

    # ---------------------------
    # Rota Çözümleme: Kullanıcının seçtiği yönteme göre
    # ---------------------------
    method = st.sidebar.selectbox("Çözüm Yöntemi Seçiniz:", ["Heuristic (Hızlı)", "MILP (Optimal Ama Yavaş)", "Heuristic + Simulated Annealing (SA)"])
    
    if method == "Heuristic + Simulated Annealing (SA)":
        solver = AdvancedVRPSolver(st.session_state.data_for_vrp)
        route_costs, _ = solver.solve_vrp_heuristic_with_sa(st.session_state.vehicles_df, forbidden_groups=st.session_state.forbidden_groups)
        
        # Calculate total cost
        total_cost = sum(rdata['vehicle']['fixed_cost'] + rdata['vehicle']['cost_per_km'] * rdata['distance']
                        for rdata in route_costs.values())
        
        st.header("Rotalama Sonuçları (Heuristic + Simulated Annealing)")
        st.metric("Toplam Maliyet", f"{total_cost:.2f} TL")
        with st.expander("Rota Detayları"):
            for vid, rdata in route_costs.items():
                route_ids = [str(solver.data.iloc[node]['ID']) for node in rdata['route']]
                st.write(f"Araç {vid}: Rota (ID'ler) -> {route_ids} | Mesafe: {rdata['distance']:.2f} km | Toplam Talep: {rdata['demand']:.2f}")
        route_map = solver.create_advanced_route_map(route_costs, st.session_state.data_for_vrp)
        folium_static(route_map, width=1000, height=600)



    elif method == "MILP (Optimal Ama Yavaş)":
        routes, total_cost = AdvancedVRPSolver(st.session_state.data_for_vrp).solve_vrp_milp(st.session_state.vehicles_df, 
                                                                                             time_limit=st.sidebar.number_input("MILP Zaman Limiti (saniye):", min_value=1, value=600, step=1))
        if routes is not None:
            st.header("Rotalama Sonuçları (MILP)")
            st.metric("Toplam Maliyet", f"{total_cost:.2f} TL")
            with st.expander("Rota Detayları"):
                for vid, route in routes.items():
                    if route:
                        route_ids = [str(AdvancedVRPSolver(st.session_state.data_for_vrp).data.iloc[node]['ID']) for node in route]
                        st.write(f"Araç {vid}: Ziyaret Sırası (ID'ler) -> {route_ids}")
            route_map = AdvancedVRPSolver(st.session_state.data_for_vrp).create_advanced_route_map(
                {vid: {'route': route,
                       'distance': compute_route_distance(route, AdvancedVRPSolver(st.session_state.data_for_vrp).dist_matrix),
                       'cost': None,
                       'vehicle': {}}
                 for vid, route in routes.items()},
                st.session_state.data_for_vrp
            )
            folium_static(route_map, width=1000, height=600)
        else:
            st.error("MILP çözümü bulunamadı.")
    else:
        route_costs, _ = AdvancedVRPSolver(st.session_state.data_for_vrp).solve_vrp_heuristic(st.session_state.vehicles_df, 
                                                                                              forbidden_groups=st.session_state.forbidden_groups)
        total_cost = sum(rdata['vehicle']['fixed_cost'] + rdata['vehicle']['cost_per_km'] * rdata['distance']
                         for rdata in route_costs.values())
        st.header("Rotalama Sonuçları (Heuristic)")
        st.metric("Toplam Maliyet", f"{total_cost:.2f} TL")
        with st.expander("Rota Detayları"):
            for vid, rdata in route_costs.items():
                if isinstance(rdata['route'], list) and len(rdata['route']) > 0 and isinstance(rdata['route'][0], list):
                    route_ids = [ [str(AdvancedVRPSolver(st.session_state.data_for_vrp).data.iloc[node]['ID']) for node in r] for r in rdata['route'] ]
                elif isinstance(rdata['route'], list):
                    route_ids = [str(AdvancedVRPSolver(st.session_state.data_for_vrp).data.iloc[node]['ID']) for node in rdata['route']]
                else:
                    route_ids = []
                st.write(f"Araç {vid}: Rota (ID'ler) -> {route_ids} | Mesafe: {rdata['distance']:.2f} km | Toplam Talep: {rdata['demand']:.2f}")
        route_map = AdvancedVRPSolver(st.session_state.data_for_vrp).create_advanced_route_map(route_costs, st.session_state.data_for_vrp)
        folium_static(route_map, width=1000, height=600)

else:
    st.warning("Lütfen hem lokasyon/tip bilgilerini hem de araç bilgilerini içeren Excel dosyalarını yükleyiniz.")

# ---------------------------
# Rota Yenileme Butonu: Node yönetimi sonrası güncel rotayı oluşturur.
if st.button("Rota Yenile"):
    solver = AdvancedVRPSolver(st.session_state.data_for_vrp)
    # Heuristic yöntemi kullanılarak yeniden rota oluşturuluyor.
    route_costs, _ = solver.solve_vrp_heuristic(st.session_state.vehicles_df, forbidden_groups=st.session_state.forbidden_groups)
    st.session_state.last_route_costs = route_costs
    updated_map = solver.create_advanced_route_map(route_costs, st.session_state.data_for_vrp)
    folium_static(updated_map, width=1000, height=600)
    st.success("Rota güncellendi!")