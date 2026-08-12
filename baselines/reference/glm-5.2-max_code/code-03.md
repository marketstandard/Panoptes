```tsx
import { useEffect, useRef, useState } from "react";

interface CountdownTimerProps {
  seconds: number;
}

function format(total: number): string {
  const clamped = Math.max(0, Math.floor(total));
  const m = Math.floor(clamped / 60);
  const s = clamped % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export default function CountdownTimer({ seconds }: CountdownTimerProps) {
  const [remaining, setRemaining] = useState(() => Math.max(0, Math.floor(seconds)));
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    setRemaining(Math.max(0, Math.floor(seconds)));

    intervalRef.current = setInterval(() => {
      setRemaining((prev) => {
        if (prev <= 1) {
          if (intervalRef.current !== null) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
          }
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [seconds]);

  return <div aria-live="polite">{format(remaining)}</div>;
}
```
