# Change log

## ogbench 1.2.1 (2026-01-14)
- Make it compatible with the latest version of `numpy` (2.0.0+).

## ogbench 1.2.0 (2025-10-20)
- Make `singletask` environments compute rewards based on `s` instead of `s'` for an `(s, a, s')` tuple.
See [this discussion](README.md/#caveats).

## ogbench 1.1.5 (2025-07-02)
- Make locomotion environments compatible with the headless mode.

## ogbench 1.1.4 (2025-06-17)
- Fix the black rendering issue in locomotion environments.

## ogbench 1.1.3 (2025-06-03)
- Add the `cube-octuple` task.

## ogbench 1.1.2 (2025-03-30)
- Improve compatibility with `gymnasium`.

## ogbench 1.1.1 (2025-03-02)
- Make it compatible with the latest version of `gymnasium` (1.1.0).

## ogbench 1.1.0 (2025-02-13)
- Added `-singletask` environments for standard (i.e., non-goal-conditioned) offline RL.
- Added `-oraclerep` environments for offline goal-conditioned RL with oracle goal representations.

## ogbench 1.0.1 (2024-10-28)
- Fixed a bug in the reward function of manipulation tasks.

## ogbench 1.0.0 (2024-10-25)
- Initial release.