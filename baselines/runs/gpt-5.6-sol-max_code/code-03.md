```tsx
import { useEffect, useState } from "react";

interface CountdownTimerProps {
  seconds: number;
}

export function CountdownTimer({ seconds }: CountdownTimerProps) {
  const [remaining, setRemaining] = useState(
    Math.max(0, Math.floor(seconds)),
  );

  useEffect(() => {
    const initial = Math.max(0, Math.floor(seconds));
    setRemaining(initial);

    if (initial === 0) {
      return;
    }

    const timerId = window.setInterval(() => {
      setRemaining((current) => {
        if (current <= 1) {
          window.clearInterval(timerId);
          return 0;
        }
        return current - 1;
      });
    }, 1000);

    return () => window.clearInterval(timerId);
  }, [seconds]);

  const minutes = Math.floor(remaining / 60);
  const displaySeconds = remaining % 60;

  return (
    <span>
      {String(minutes).padStart(2, "0")}:
      {String(displaySeconds).padStart(2, "0")}
    </span>
  );
}
```