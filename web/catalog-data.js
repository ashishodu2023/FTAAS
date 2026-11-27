/** Offline copy of configs/settings.yaml lists and the sentences GET /v1/catalog returns. */

export const STATIC_CATALOG = {
  frameworks: ["transformers", "trl", "verl", "llama-factory", "unsloth", "axolotl"],
  techniques: {
    full_finetune: ["16bit_full", "frozen"],
    peft: [
      "lora",
      "qlora",
      "dora",
      "lora_plus",
      "longlora",
      "loftq",
      "pissa",
      "prefix_tuning",
      "adapter_tuning",
      "bitfit",
      "ia3",
    ],
    instruction_tuning: ["sft", "multimodal_instruction", "self_instruct"],
    alignment: ["reward_modeling", "ppo", "dpo", "orpo"],
  },
  defaults: {
    framework: "transformers",
    technique: "lora",
    model_name: "distilgpt2",
    hyperparameters: {
      learning_rate: 0.001,
      max_steps: 24,
      lora_r: 8,
      max_seq_length: 64,
    },
  },
  backends: {
    transformers: "live — PEFT/transformers Trainer; QLoRA when CUDA+bnb",
    trl: "live — TRL SFT/DPO when installed; else PEFT",
    verl: "compat — PEFT/TRL backend",
    "llama-factory": "compat — PEFT backend",
    unsloth: "compat — PEFT backend",
    axolotl: "compat — PEFT backend",
  },
  inference: {
    transformers: "live — local HF/PEFT generate",
    vllm: "planned",
    ray_serve: "planned",
  },
  phases: [
    { phase: 0, name: "Fine-Tuning & RL Templates", status: "available" },
    { phase: 1, name: "Fine-Tuning UI", status: "available" },
    { phase: 2, name: "Fine-Tune & Evaluate", status: "available" },
    { phase: 3, name: "Resource Optimization", status: "planned" },
    { phase: 4, name: "Sweeps & Optimization", status: "planned" },
  ],
  training: {
    mode: "local",
    qlora: {
      requires: "CUDA + bitsandbytes (see Dockerfile.gpu / docs/gpu-training.md)",
      available: false,
    },
  },
};

export function mergeCatalog(live) {
  if (!live || typeof live !== "object") {
    return { ...STATIC_CATALOG, source: "static" };
  }
  return {
    ...STATIC_CATALOG,
    ...live,
    backends: { ...STATIC_CATALOG.backends, ...(live.backends || {}) },
    inference: { ...STATIC_CATALOG.inference, ...(live.inference || {}) },
    defaults: { ...STATIC_CATALOG.defaults, ...(live.defaults || {}) },
    source: "gateway",
  };
}
