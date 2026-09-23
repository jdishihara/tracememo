# Normalized table schemas

Adapters write these tables; analyses read them. All times are `t_s`, seconds from
arming (float64). Positions are metres in a local level frame (x east, y north, z up
unless the adapter config says otherwise). Angles are radians; `yaw_rad` is wrapped to
(-pi, pi]. ArduPilot logs are NED; the `ardupilot` adapter's default profile converts to ENU.

| Table | Columns | Notes |
|---|---|---|
| `pose` | `t_s, x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad` | Vehicle's own position/attitude estimate |
| `nav_target` | `t_s, x_m, y_m, z_m` | Commanded / planned position |
| `beacon` | `t_s, x_m, y_m, z_m, quality` | External positioning (e.g. ultrasonic beacons); `quality` optional, adapter-defined |
| `beacon_raw` | `t_s, x_m, y_m, z_m, quality` | Marvelmind export via the `marvelmind` adapter (`options.table` renames it, e.g. to `beacon`) |
| `spans` | `trace_id, span_id, parent_id, name, start, end, duration_ms, metadata` | LLM pipeline spans (milestone 5) |
| `traces` | `trace_id, version, total_latency_ms, input, output` | LLM traces (milestone 5) |
| `eval_scores` | `item_id, config, metric_name, score` | Long-format eval results (milestone 5) |
