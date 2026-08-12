```tsx
import { useEffect, useState } from "react";

export interface CountdownTimerProps {
  /** Starting duration in seconds. Negative or fractional values are floored to a sane whole number. */
  seconds: number;
  /** Optional callback fired once the timer reaches 00:00. */
  onComplete?: () => void;
}

function formatTime(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const secs = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

export function CountdownTimer({ seconds, onComplete }: CountdownTimerProps) {
  const [remaining, setRemaining] = useState(() => Math.max(0, Math.floor(seconds)));

  // Restart whenever the caller supplies a new duration.
  useEffect(() => {
    setRemaining(Math.max(0, Math.floor(seconds)));
  }, [seconds]);

  const isRunning = remaining > 0;

  useEffect(() => {
    if (!isRunning) return;

    const id = setInterval(() => {
      setRemaining((prev) => Math.max(0, prev - 1));
    }, 1000);

    return () => clearInterval(id);
  }, [isRunning]);

  useEffect(() => {
    if (remaining === 0) onComplete?.();
  }, [remaining, onComplete]);

  return (
    <span role="timer" aria-live="off">
      {formatTime(remaining)}
    </span>
  );
}

export default CountdownTimer;
```

A few decisions worth flagging:

- **Cleanup.** The interval is cleared in the effect's teardown, which covers unmount, a change in `seconds`, and the transition to zero. Keying the effect on the boolean `isRunning` rather than on `remaining` means the interval is created once for the whole countdown instead of being torn down and rebuilt every tick.
- **Functional state update.** `setRemaining((prev) => ...)` avoids capturing a stale `remaining` in the interval closure, which is the classic bug in this component.
- **Formatting past an hour.** `mm` is not capped at 59, so 3700 seconds renders as `61:40`. If you want `HH:MM:SS` past the hour mark, that's a change to `formatTime` only.
- **Drift.** `setInterval` accumulates a little error and browsers throttle timers in background tabs. If the displayed value has to stay accurate over long durations, store a target `Date.now() + seconds * 1000` and compute the remainder from the wall clock on each tick instead of decrementing.
