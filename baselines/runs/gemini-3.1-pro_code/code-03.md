```tsx
import React, { useState, useEffect } from 'react';

interface CountdownTimerProps {
  seconds: number;
}

export const CountdownTimer: React.FC<CountdownTimerProps> = ({ seconds }) => {
  const [timeLeft, setTimeLeft] = useState(seconds);

  useEffect(() => {
    if (timeLeft <= 0) return;

    const timerId = setInterval(() => {
      setTimeLeft(prev => prev - 1);
    }, 1000);

    return () => clearInterval(timerId);
  }, [timeLeft]);

  const minutes = Math.floor(timeLeft / 60);
  const remainingSeconds = timeLeft % 60;
  
  const formattedMinutes = minutes.toString().padStart(2, '0');
  const formattedSeconds = remainingSeconds.toString().padStart(2, '0');

  return (
    <div>
      {formattedMinutes}:{formattedSeconds}
    </div>
  );
};
```