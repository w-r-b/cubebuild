"""层构建器注册表。

每个已实现构建器的层在此登记；后续新增构建器时在此挂载。
未登记的层由构建入口跳过并记入 pending（骨架阶段预期行为）；
已登记但源数据未集齐的层（如 SourceNotReady）同样跳过并提示原因。
"""

from .contract import LayerEntry
from .derived import build_landsea_mask, build_relief, build_slope
from .etopo import build_etopo_bedrock
from .faults import build_gem_fault_distance, build_gem_fault_slip_class
from .gravmag import build_emag2_sealevel, build_wgm2012_bouguer
from .gsrm import build_gsrm_strain_layer
from .landcat import (
    build_crustal_age_class,
    build_glim_lithology_class,
    build_gum_lithology_class,
    build_gum_thickness_class,
)
from .litho import (
    build_gemma_basement_depth,
    build_gemma_moho,
    build_gemma_moho_err,
    build_litho1_lab,
    build_seafloor_age,
)
from .paleo import build_paleodem_elevation
from .pb2002 import (
    build_pb2002_boundary_distance,
    build_pb2002_plate_id,
    build_pb2002_zone_class,
)
from .pixel_area import build_pixel_area
from .points import build_gsrm_gps_density, build_wsm_density
from .seismic import build_glad_layer, build_semucb_layer
from .sediment import (
    build_global_basins_class,
    build_global_basins_distance,
    build_gst1_thickness,
)
from .slab2 import build_slab2_data_layer, build_slab2_overlap_mask
from .thermal import build_hfgrid14_heatflow, build_heatflow_density
from .vector import build_hasterok_province_class

# 层 id → 构建器 (entry, tier) -> xr.DataArray
# 3D 地震层返回 [depth, lat, lon] 数组，cli 按 SEISMIC_3D_GROUP_OF
# 路由入档位子节点（深度坐标 289/74 互不相同，不能共存于档位节点）。
# 4D 古高程层返回 [time, lat, lon] 数组，cli 按 PALEO_4D_GROUP_OF
# 路由入 /{tier}/paleodem（与档位节点 2D 层形状不同，同节点严格配准约束）。
LAYER_BUILDERS = {
    "derived__pixel_area": build_pixel_area,
    "derived__landsea_mask": build_landsea_mask,
    "derived__slope": build_slope,
    "derived__relief": build_relief,
    "topography__bedrock_elevation": build_etopo_bedrock,
    "gravity__wgm2012_bouguer": build_wgm2012_bouguer,
    "magnetics__emag2_sealevel": build_emag2_sealevel,
    "lithosphere__hasterok_province_class": build_hasterok_province_class,
    "stress_kinematics__wsm_density": build_wsm_density,
    "stress_kinematics__gsrm_gps_density": build_gsrm_gps_density,
    "stress_kinematics__gem_fault_slip_class": build_gem_fault_slip_class,
    "stress_kinematics__gem_fault_distance": build_gem_fault_distance,
    "stress_kinematics__pb2002_plate_id": build_pb2002_plate_id,
    "stress_kinematics__pb2002_zone_class": build_pb2002_zone_class,
    "stress_kinematics__pb2002_boundary_distance": build_pb2002_boundary_distance,
    "stress_kinematics__gsrm_exx": build_gsrm_strain_layer,
    "stress_kinematics__gsrm_eyy": build_gsrm_strain_layer,
    "stress_kinematics__gsrm_exy": build_gsrm_strain_layer,
    "stress_kinematics__gsrm_vorticity": build_gsrm_strain_layer,
    "stress_kinematics__slab2_depth": build_slab2_data_layer,
    "stress_kinematics__slab2_dip": build_slab2_data_layer,
    "stress_kinematics__slab2_strike": build_slab2_data_layer,
    "stress_kinematics__slab2_thickness": build_slab2_data_layer,
    "stress_kinematics__slab2_uncertainty": build_slab2_data_layer,
    "stress_kinematics__slab2_overlap_mask": build_slab2_overlap_mask,
    "thermal__heatflow_density": build_heatflow_density,
    "thermal__hfgrid14_heatflow": build_hfgrid14_heatflow,
    "seismology__gladm35_vsv": build_glad_layer,
    "seismology__gladm35_vsh": build_glad_layer,
    "seismology__gladm35_vpv": build_glad_layer,
    "seismology__gladm35_vph": build_glad_layer,
    "seismology__gladm35_eta": build_glad_layer,
    "seismology__semucb_vs": build_semucb_layer,
    "seismology__semucb_xi": build_semucb_layer,
    "paleo__paleodem_elevation": build_paleodem_elevation,
    "lithosphere__glim_lithology_class": build_glim_lithology_class,
    "sediment__gum_lithology_class": build_gum_lithology_class,
    "sediment__gum_thickness_class": build_gum_thickness_class,
    "lithosphere__crustal_age_class": build_crustal_age_class,
    "lithosphere__gemma_moho": build_gemma_moho,
    "lithosphere__gemma_moho_err": build_gemma_moho_err,
    "lithosphere__gemma_basement_depth": build_gemma_basement_depth,
    "lithosphere__litho1_lab": build_litho1_lab,
    "lithosphere__seafloor_age": build_seafloor_age,
    "sediment__gst1_thickness": build_gst1_thickness,
    "sediment__global_basins_class": build_global_basins_class,
    "sediment__global_basins_distance": build_global_basins_distance,
}
