-- custom_car.lua
-- OSRM car profile with traffic signals & traffic calming penalties, plus
-- more realistic turn/time weighting.

api_version = 4

-- OSRM 라이브러리
Set = require('lib/set')
Sequence = require('lib/sequence')
Handlers = require('lib/way_handlers')
Relations = require('lib/relations')
Obstacles = require("lib/obstacles")
Utils = require("lib/utils")
Measure = require("lib/measure")
limit = require("lib/maxspeed").limit
find_access_tag = require("lib/access").find_access_tag

-- 1) 프로필 초기 설정
function setup()
  return {
    properties = {
      weight_name                     = 'duration',   -- speed/turn/penalty 기반으로 'duration'(시간) 우선
      max_speed_for_map_matching      = 180/3.6,
      process_call_tagless_node      = false,
      u_turn_penalty                 = 120,   -- 유턴 시 큰 페널티 (도심에서 유턴 회피,초 단위 추가)
      continue_straight_at_waypoint  = true,
      use_turn_restrictions          = true,
      left_hand_driving              = false,
    },

    default_mode              = mode.driving,
    default_speed             = 10,    -- (km/h)
    oneway_handling           = true,
    side_road_multiplier      = 0.8,
    turn_penalty              = 18.0,  -- 회전 페널티 (기존보다 크게)
    speed_reduction           = 0.8,
    turn_bias                 = 1.275,
    cardinal_directions       = false,

    vehicle_height = 2.0,
    vehicle_width  = 1.9,
    vehicle_length = 4.8,
    vehicle_weight = 2000,

    suffix_list = {
      'N','NE','E','SE','S','SW','W','NW','North','South','West','East','Nor','Sou','We','Ea'
    },

    -- 진입 가능/불가능 판정
    barrier_whitelist = Set {
      'cattle_grid','border_control','toll_booth','sally_port','gate','lift_gate','no','entrance','height_restrictor','arch'
    },
    access_tag_whitelist = Set {
      'yes','motorcar','motor_vehicle','vehicle','permissive','designated','hov'
    },
    access_tag_blacklist = Set {
      'no','agricultural','forestry','emergency','psv','customers','private','delivery','destination'
    },
    service_access_tag_blacklist = Set {'private'},
    restricted_access_tag_list = Set {'private','delivery','destination','customers'},
    access_tags_hierarchy = Sequence {'motorcar','motor_vehicle','vehicle','access'},
    service_tag_forbidden = Set {'emergency_access'},
    restrictions = Sequence {'motorcar','motor_vehicle','vehicle'},

    -- OSRM classes
    classes = Sequence {'toll','motorway','ferry','restricted','tunnel'},
    excludable = Sequence { Set {'toll'}, Set {'motorway'}, Set {'ferry'} },
    avoid = Set {'area','reversible','impassable','hov_lanes','steps','construction','proposed'},

    -- 2) 도로별 속도 km/h (주요)
    speeds = Sequence {
      highway = {
        motorway = 90, motorway_link = 45,
        trunk = 85, trunk_link = 40,
        primary = 65, primary_link = 30,
        secondary = 55, secondary_link = 25,
        tertiary = 40, tertiary_link = 20,
        unclassified = 25, residential = 25,
        living_street = 10, service = 15
      }
    },

    service_penalties = {
      alley = 0.5, parking = 0.5, parking_aisle = 0.5,
      driveway = 0.5, ["drive-through"] = 0.5, ["drive-thru"] = 0.5
    },

    restricted_highway_whitelist = Set {
      'motorway','motorway_link','trunk','trunk_link','primary','primary_link','secondary','secondary_link','tertiary','tertiary_link','residential','living_street','unclassified','service'
    },
    construction_whitelist = Set {'no','widening','minor'},

    route_speeds = { ferry = 5, shuttle_train = 10 },
    bridge_speeds = { movable = 5 },

    -- 3) 도로 표면 속도 보정
    surface_speeds = {
      asphalt = nil, concrete = nil, ["concrete:plates"] = nil, ["concrete:lanes"] = nil, paved = nil,
      cement = 80, compacted = 80, fine_gravel = 80,
      paving_stones = 60, metal = 60, bricks = 60,
      grass = 40, wood = 40, sett = 40, grass_paver = 40, gravel = 40, unpaved = 40, ground = 40, dirt = 40, pebblestone = 40, tartan = 40,
      cobblestone = 30, clay = 30, earth = 20, stone = 20, rocky = 20, sand = 20, mud = 10
    },
    tracktype_speeds = { grade1 = 60, grade2 = 40, grade3 = 30, grade4 = 25, grade5 = 20 },
    smoothness_speeds = { intermediate = 80, bad = 40, very_bad = 20, horrible = 10, very_horrible = 5, impassable = 0 },

    -- 4) 지역별 제한속도 테이블
    maxspeed_table_default = { urban = 50, rural = 90, trunk = 110, motorway = 130 },
    maxspeed_table = {
      ["at:rural"] = 100, ["at:trunk"] = 100, ["be:motorway"] = 120, ["be-bru:rural"] = 70, ["be-bru:urban"] = 30,
      ["be-vlg:rural"] = 70, ["bg:motorway"] = 140, ["by:urban"] = 60, ["by:motorway"] = 110, ["ca-on:rural"] = 80,
      ["ch:rural"] = 80, ["ch:trunk"] = 100, ["ch:motorway"] = 120, ["cz:trunk"] = 0, ["cz:motorway"] = 0,
      ["de:living_street"] = 7, ["de:rural"] = 100, ["de:motorway"] = 0, ["dk:rural"] = 80, ["es:trunk"] = 90,
      ["fr:rural"] = 80, ["gb:nsl_single"] = (60*1609)/1000, ["gb:nsl_dual"] = (70*1609)/1000, ["gb:motorway"] = (70*1609)/1000,
      ["nl:rural"] = 80, ["nl:trunk"] = 100, ["no:rural"] = 80, ["no:motorway"] = 110, ["ph:urban"] = 40,
      ["ph:rural"] = 80, ["ph:motorway"] = 100, ["pl:rural"] = 100, ["pl:expressway"] = 120, ["pl:motorway"] = 140,
      ["ro:trunk"] = 100, ["ru:living_street"] = 20, ["ru:urban"] = 60, ["ru:motorway"] = 110,
      ["uk:nsl_single"] = (60*1609)/1000, ["uk:nsl_dual"] = (70*1609)/1000, ["uk:motorway"] = (70*1609)/1000,
      ['za:urban'] = 60, ['za:rural'] = 100, ["none"] = 140
    },

    -- 5) 기타
    relation_types = Sequence { "route" },
    highway_turn_classification = {},
    access_turn_classification  = {}
  }
end

-- 6) 노드(신호등) 처리: node에 highway=traffic_signals 있으면 신호등 플래그 저장
function process_node(profile, node, result)
  -- local highway = node:get_value_by_key("highway")
  -- if highway == "traffic_signals" then
  --   result.traffic_light = true
  -- end
end

-- 7) 도로(방지턱) 처리: way에 traffic_calming=* 있으면 방지턱 플래그 저장
function process_way(profile, way, result)
  Handlers.run(profile, way, result)

  local calming = way:get_value_by_key("traffic_calming")
  if calming and calming ~= "" and calming ~= "no" then
    result.traffic_calming = true
  end
end

-- 8) 회전 처리: 회전각도 + 신호등/방지턱 패널티 반영
function process_turn(profile, turn)
  local turn_penalty = profile.turn_penalty
  local turn_bias = turn.is_left_hand_driving and 1. / profile.turn_bias or profile.turn_bias
  local base_penalty = turn_penalty

  -- 도로 등급별 패널티 (예: 고속도로 vs residential 등)
  local classification = turn.source_road.classification or ""
  local target_classification = turn.target_road.classification or ""

  if classification == "primary" or classification == "trunk" or classification == "motorway" then
    base_penalty = base_penalty * 1.5
  elseif classification == "residential" or target_classification == "residential" then
    base_penalty = base_penalty * 0.2
  elseif classification == "secondary" or target_classification == "secondary" then
    base_penalty = base_penalty * 1.2
  elseif classification == "primary" or target_classification == "primary" then
    base_penalty = base_penalty * 1.4
  end

  -- 기본 회전 각도 페널티
  if turn.angle >= 0 then
    turn.duration = turn.duration + (base_penalty * 0.7) 
                    / (1 + math.exp(-((13 / turn_bias) * turn.angle / 180 - 6.5 * turn_bias)))
  else
    turn.duration = turn.duration + (base_penalty * 1.3) 
                    / (1 + math.exp(-((13 * turn_bias) * -turn.angle / 180 - 6.5 / turn_bias)))
  end

  -- 신호등 패널티 (node에서 setting한 값)
  if turn.has_traffic_light then
    turn.duration = turn.duration + 10.0  -- 예: 10초
  end

  -- 방지턱 패널티 (way에서 setting한 값)
  if turn.source_road and turn.source_road.traffic_calming then
    turn.duration = turn.duration + 5.0   -- 예: 5초
  end

  -- U턴 패널티
  if turn.is_u_turn then
    turn.duration = turn.duration + profile.properties.u_turn_penalty
  end

  turn.weight = turn.duration

  -- 디버그 출력
  print(string.format(
    "Turn %s → %s | angle: %.1f° | duration: %.2fs | weight: %.2f%s%s",
    tostring(turn.from), tostring(turn.via), turn.angle, turn.duration, turn.weight,
    turn.has_traffic_light and " [traffic_light]" or "",
    (turn.source_road and turn.source_road.traffic_calming) and " [traffic_calming]" or ""
  ))

  -- weight_name='duration'이면 turn.weight는 duration
end

return {
  setup        = setup,
  process_way  = process_way,
  process_node = process_node,
  process_turn = process_turn
}
