import { render, screen } from '@testing-library/react';
import { expect, test } from 'vitest';
import App from './App';

test('renders empty state and input controls', () => {
  render(<App />);
  expect(screen.getByText(/Analyze AI participation/i)).toBeTruthy();
  expect(screen.getByLabelText(/Text or code to analyze/i)).toBeTruthy();
  expect(screen.getByText(/Start with a sample/i)).toBeTruthy();
});
