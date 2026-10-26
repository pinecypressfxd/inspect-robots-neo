# inspect-robots-harness

HarnessVLA policy for [Inspect Robots](https://github.com/robocurve/inspect-robots).
Registered as `harness`.

The planner is the same Astra chat client as `hybrid`. Motion goes through the
hybrid executor's existing interfaces:

- `move_to`, `set_gripper`: analytic end-effector targets on the embodiment's
  dimension labels.
- `release`: open one gripper to 0.09.
- `vla_act`: one red-cup grasp. The runtime sends the vla plugin's trained
  sentence. The attempt runs until `max_seconds` or a tracking interrupt.
- `remember`: append a fact to the trial memory attached to later observations.

A repeat `vla_act` is refused until `move_to`, `set_gripper`, or `release` has
restaged the arm. After `max_vla_attempts` (default 3) the red cup is grasped
with `set_gripper`. A goal that does not contain `red cup` has no `vla_act`.

```bash
inspect-robots "put the red cup in the basket" \
  --policy harness -P model=mog-10-a \
  -P wire=responses \
  -P base_url=https://kspmas.ksyun.com/v1 \
  -P api_key_env=JINSHANYUN_CHATGPT_KEY \
  --embodiment nero -E control_hz=10 -P control_hz=10
```
