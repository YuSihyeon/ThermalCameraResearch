import open3d as o3d
import numpy as np
import pandas as pd

def generate_mock_radar_data():
    """
    실제 CSV 데이터셋이 없을 때 즉시 테스트해볼 수 있도록
    방(Room) 형태의 가상 레이더 포인트 클라우드를 생성합니다.
    (실제 데이터 적용 시 이 함수 대신 pandas로 csv를 읽으세요)
    """
    # 바닥, 3면의 벽, 노이즈(연기), 사람(장애물) 생성
    floor = np.random.rand(1000, 3) * [5, 5, 0.1]
    wall1 = np.random.rand(800, 3) * [0.1, 5, 2.5]
    wall2 = np.random.rand(800, 3) * [5, 0.1, 2.5]
    wall2[:, 1] += 5.0
    wall3 = np.random.rand(800, 3) * [5, 0.1, 2.5]
    
    noise = np.random.rand(1500, 3) * [5, 5, 2.5] # 짙은 연기로 인한 산란 노이즈
    human = np.random.rand(200, 3) * [0.5, 0.5, 1.8] + [2.5, 2.5, 0] # 방 한가운데 사람
    
    data = np.vstack((floor, wall1, wall2, wall3, noise, human))
    return data

def process_radar_data(points):
    # 1. Open3D 포인트 클라우드 객체로 변환
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5]) # 초기 색상: 회색
    print(f"초기 레이더 포인트 수: {len(pcd.points)}")

    # 2. 노이즈 제거 (Statistical Outlier Removal)
    # 농연(Smoke)이나 열기로 인해 허공에 뜨는 레이더 고스트 포인트를 제거합니다.
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    pcd_clean = pcd.select_by_index(ind)
    print(f"노이즈 필터링 후 포인트 수: {len(pcd_clean.points)}")

    # 3. 바닥 평면 검출 및 분리 (RANSAC 알고리즘)
    # 소방관이 걷는 바닥을 먼저 인식하여 맵에서 제외합니다.
    plane_model_floor, inliers_floor = pcd_clean.segment_plane(distance_threshold=0.15,
                                                               ransac_n=3,
                                                               num_iterations=1000)
    floor_cloud = pcd_clean.select_by_index(inliers_floor)
    floor_cloud.paint_uniform_color([0.0, 1.0, 0.0]) # 바닥은 초록색
    
    # 바닥이 아닌 나머지 데이터 (벽, 장애물 등)
    non_floor_cloud = pcd_clean.select_by_index(inliers_floor, invert=True)

    # 4. 벽(Wall) 평면 검출 (RANSAC)
    # 남은 포인트 중 가장 큰 평면을 벽으로 인식합니다.
    plane_model_wall, inliers_wall = non_floor_cloud.segment_plane(distance_threshold=0.2,
                                                                   ransac_n=3,
                                                                   num_iterations=1000)
    wall_cloud = non_floor_cloud.select_by_index(inliers_wall)
    wall_cloud.paint_uniform_color([1.0, 0.0, 0.0]) # 벽은 빨간색

    # 벽도 아닌 나머지 데이터 (요구조자, 장애물 등)
    obstacles_cloud = non_floor_cloud.select_by_index(inliers_wall, invert=True)
    obstacles_cloud.paint_uniform_color([0.0, 0.0, 1.0]) # 장애물은 파란색

    # 5. 결과 3D 렌더링 (AR 글래스에 투사될 정보의 기초)
    print("--- 3D 시각화 창이 열립니다. 마우스로 드래그하여 확인하세요 ---")
    o3d.visualization.draw_geometries([floor_cloud, wall_cloud, obstacles_cloud],
                                      window_name="FireSight AR - Radar Perception",
                                      width=1024, height=768)

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    
    print("파일 찾기를 생략하고, 데이터를 직접 주입하여 테스트를 시작합니다...")
    
    # 1. 파일에 저장하는 대신, 코드 안에서 즉석으로 데이터를 생성합니다.
    floor_x = np.random.uniform(-4, 4, 1000)
    floor_y = np.random.uniform(0, 8, 1000)
    floor_z = np.random.normal(0, 0.05, 1000)
    
    wall_x = np.random.uniform(-4, 4, 800)
    wall_y = np.random.normal(8, 0.05, 800)
    wall_z = np.random.uniform(0, 3, 800)
    
    obs_x = np.random.normal(0, 0.3, 300)
    obs_y = np.random.normal(4, 0.3, 300)
    obs_z = np.random.uniform(0, 1.8, 300)
    
    x = np.concatenate([floor_x, wall_x, obs_x])
    y = np.concatenate([floor_y, wall_y, obs_y])
    z = np.concatenate([floor_z, wall_z, obs_z])
    
    # 2. pd.read_csv()로 엑셀을 읽어온 것과 100% 완벽하게 동일한 상태로 만듭니다.
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    
    # 3. 좌표만 쏙 뽑아서 3D 시각화 신경망에 던져줍니다!
    real_points = df[['x', 'y', 'z']].values
    process_radar_data(real_points)