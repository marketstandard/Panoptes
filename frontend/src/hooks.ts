import { useEffect, useState } from 'react';
import { fetchArtifact, type ArtifactName } from './api';

export function useArtifact<T>(name: ArtifactName): { data: T | null; loading: boolean } {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let cancelled = false;
    fetchArtifact<T>(name)
      .then((payload) => {
        if (!cancelled) {
          setData(payload);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setData(null);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [name]);
  return { data, loading };
}
