import { useState, useCallback } from 'react';
import { Model } from '../api/types';

export interface ModelOption {
  value: string;
  label: string;
  model_name: string;
}

export const MODEL_OPTIONS: ModelOption[] = [
  { value: 'openai-gpt-4o-mini', label: 'OpenAI GPT-4o-mini (Recommended)', model_name: 'gpt-4o-mini' },
  { value: 'openai-gpt-4o', label: 'OpenAI GPT-4o', model_name: 'gpt-4o' },
  { value: 'openai-gpt-4', label: 'OpenAI GPT-4', model_name: 'gpt-4' },
  { value: 'openai-gpt-3.5-turbo', label: 'OpenAI GPT-3.5 Turbo', model_name: 'gpt-3.5-turbo' },
];

const DEFAULT_MODEL: Model = {
  name: 'openai-gpt-4o-mini',
  description: 'OpenAI GPT-4o-mini',
  model_name: 'gpt-4o-mini',
};

interface UseModelSelectorOptions {
  defaultDescription?: string;
}

export function useModelSelector(options: UseModelSelectorOptions = {}) {
  const { defaultDescription = 'OpenAI GPT-4o-mini' } = options;

  const [model, setModel] = useState<Model>({
    ...DEFAULT_MODEL,
    description: defaultDescription,
  });

  const handleModelChange = useCallback((value: string) => {
    const selected = MODEL_OPTIONS.find((opt) => opt.value === value);
    if (selected) {
      setModel((prev) => ({
        ...prev,
        name: selected.value,
        model_name: selected.model_name,
      }));
    }
  }, []);

  return {
    model,
    setModel,
    handleModelChange,
    modelOptions: MODEL_OPTIONS,
  };
}
