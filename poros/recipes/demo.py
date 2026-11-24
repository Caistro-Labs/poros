"""The shared set a preview shows (POROS_DEV_SIMULATE): what a recipes repo holds, every file read
through ``rules.parse`` as a fetched one is. People, numbers and dates are made up for the preview:
the card says "demo" beside the repo, and none of it is ever written anywhere."""

from __future__ import annotations

from typing import List, Tuple


def _f(key: str, body: str) -> Tuple[str, str]:
    return key, body.strip() + "\n"


FILES: List[Tuple[str, str]] = [
    # -- big models: the exact settings that make one train on one card -------------------------
    _f("qwen2.5-72b-4090", '''
title = "Qwen2.5-72B on one 4090"
about = "The 72B in 24 GB: the NF4 base streamed four blocks at a time, rank 16, 512 tokens"
by = "kestrel-dev"
use = "big"
updated = 2026-11-08
model = "Qwen/Qwen2.5-72B-Instruct"
data = "chats"
[settings]
max_seq_length = 512
block_size = 4
lora_rank = 16
lora_alpha = 32
learning_rate = 1e-4
gradient_accumulation_steps = 8
warmup_steps = 20
'''),
    _f("qwen3-235b-a22b-pro6000", '''
title = "Qwen3-235B-A22B, one block resident"
about = "The 235B MoE on a 96 GB card: one block in memory, rank 8, 1K tokens"
by = "tensorwright"
use = "big"
updated = 2026-11-06
model = "Qwen/Qwen3-235B-A22B-Instruct-2507"
data = "chats"
[settings]
max_seq_length = 1024
block_size = 1
lora_rank = 8
learning_rate = 5e-5
gradient_accumulation_steps = 16
num_epochs = 1
[measured]
machine = "RTX PRO 6000 Blackwell Workstation Edition"
peak_gib = 88.4
s_per_step = 96.0
'''),
    _f("mistral-large-123b-h100", '''
title = "Mistral Large 123B on an H100"
about = "123B dense at 4K tokens on 80 GB; a family Poros has not validated"
by = "tinkerer-42"
use = "big"
updated = 2026-10-30
model = "mistralai/Mistral-Large-Instruct-2411"
data = "qa"
unvalidated = "mistral"
[settings]
max_seq_length = 4096
block_size = 4
lora_rank = 32
learning_rate = 1e-4
gradient_accumulation_steps = 4
[measured]
machine = "H100 80GB HBM3"
peak_gib = 71.2
s_per_step = 18.7
'''),
    _f("llama-3.3-70b-4090", '''
title = "Llama 3.3 70B on a 4090"
about = "70B on 24 GB: four blocks streamed, 1K tokens, rank 16"
by = "ali-r"
use = "big"
updated = 2026-10-21
model = "meta-llama/Llama-3.3-70B-Instruct"
data = "chats"
[settings]
max_seq_length = 1024
block_size = 4
lora_rank = 16
learning_rate = 2e-4
gradient_accumulation_steps = 8
[measured]
machine = "RTX 4090"
peak_gib = 22.6
s_per_step = 27.4
'''),
    _f("qwen2.5-72b-8k", '''
title = "Qwen2.5-72B at 8K tokens"
about = "Long documents on the 72B: 8K tokens, two blocks, an H100's 80 GB"
by = "datawhale-dan"
use = "big"
updated = 2026-10-12
model = "Qwen/Qwen2.5-72B-Instruct"
data = "text"
[settings]
max_seq_length = 8192
block_size = 2
lora_rank = 16
learning_rate = 1e-4
[measured]
machine = "H100 80GB HBM3"
peak_gib = 64.9
s_per_step = 52.3
'''),
    # -- the Mac path: settings that hold in unified memory --------------------------------------
    _f("m4-max-32b", '''
title = "M4 Max: a 32B on 128 GB"
about = "Qwen2.5-32B in 128 GB unified memory, the Mac's smaller learning rate"
by = "mira-k"
use = "mac"
updated = 2026-11-09
model = "Qwen/Qwen2.5-32B-Instruct"
data = "chats"
[settings]
max_seq_length = 1024
lora_rank = 16
learning_rate = 2e-5
num_epochs = 1
[measured]
machine = "apple-m4-max-40gpu-128gb"
peak_gib = 61.3
s_per_step = 14.8
'''),
    _f("m3-ultra-72b", '''
title = "M3 Ultra: 72B at 2K tokens"
about = "The 72B whole in 512 GB: no streaming, 2K tokens, rank 32"
by = "ines-varga"
use = "mac"
updated = 2026-11-01
model = "Qwen/Qwen2.5-72B-Instruct"
data = "chats"
[settings]
max_seq_length = 2048
lora_rank = 32
learning_rate = 2e-5
[measured]
machine = "apple-m3-ultra-80gpu-512gb"
peak_gib = 158.0
s_per_step = 33.5
'''),
    _f("air-16gb-1.5b", '''
title = "MacBook Air 16 GB: a 1.5B"
about = "The largest that trains on a 16 GB Air without swapping, 512 tokens"
by = "devon-t"
use = "mac"
updated = 2026-10-27
model = "Qwen/Qwen2.5-1.5B-Instruct"
data = "qa"
[settings]
max_seq_length = 512
lora_rank = 8
learning_rate = 2e-5
[measured]
machine = "apple-m3-10gpu-16gb"
peak_gib = 6.2
s_per_step = 3.9
'''),
    _f("m4-pro-gemma-12b", '''
title = "M4 Pro 48 GB: Gemma 3 12B"
about = "Gemma's 12B in 48 GB: 2K tokens, rank 16"
by = "jules-okafor"
use = "mac"
updated = 2026-10-18
model = "google/gemma-3-12b-it"
data = "chats"
[settings]
max_seq_length = 2048
lora_rank = 16
learning_rate = 2e-5
[measured]
machine = "apple-m4-pro-20gpu-48gb"
peak_gib = 31.4
s_per_step = 9.6
'''),
    # -- by what they're for ---------------------------------------------------------------------
    _f("bot-in-five", '''
title = "A bot in five minutes"
about = "The smallest Qwen3 on 200 of your chats: the whole loop before a long run"
by = "ines-varga"
use = "start"
updated = 2026-11-08
model = "Qwen/Qwen3-0.6B"
data = "chats"
[settings]
max_seq_length = 512
max_steps = 200
lora_rank = 16
[measured]
machine = "RTX 4090"
peak_gib = 3.1
s_per_step = 0.4
'''),
    _f("shopify-returns", '''
title = "Shopify returns desk"
about = "Return and refund answers from 3,000 resolved tickets"
by = "shopfront-labs"
use = "support"
updated = 2026-11-07
model = "Qwen/Qwen2.5-7B-Instruct"
data = "qa"
[settings]
max_seq_length = 1024
lora_rank = 16
learning_rate = 2e-4
num_epochs = 2
[measured]
machine = "RTX 4090"
peak_gib = 14.8
s_per_step = 2.1
'''),
    _f("atendimento-pt", '''
title = "Atendimento ao cliente em português do Brasil"
about = "Support in Brazilian Portuguese, formal você, from 12,000 conversations"
by = "ana-lucia-ferreira-santos"
use = "support"
updated = 2026-11-04
model = "Qwen/Qwen3-14B"
data = "chats"
[settings]
max_seq_length = 2048
lora_rank = 32
learning_rate = 1e-4
[measured]
machine = "RTX 3090"
peak_gib = 21.7
s_per_step = 6.3
'''),
    _f("it-helpdesk", '''
title = "IT helpdesk, internal"
about = "Password resets, VPN, laptop requests: short answers with links"
by = "devon-t"
use = "support"
updated = 2026-10-25
model = "Qwen/Qwen3-4B"
data = "qa"
[settings]
max_seq_length = 512
lora_rank = 16
'''),
    _f("newsletter-voice", '''
title = "Newsletter in my voice"
about = "Two years of issues: the openings, the asides, the sign-off"
by = "mira-k"
use = "writing"
updated = 2026-11-05
model = "Qwen/Qwen2.5-7B-Instruct"
data = "text"
[settings]
max_seq_length = 2048
lora_rank = 32
learning_rate = 1e-4
num_epochs = 3
[measured]
machine = "RTX 4090"
peak_gib = 17.2
s_per_step = 3.4
'''),
    _f("screenplay-dialogue", '''
title = "Screenplay dialogue"
about = "Lines that sound spoken: interruptions, subtext, no speeches"
by = "jules-okafor"
use = "writing"
updated = 2026-10-29
model = "google/gemma-3-27b-it"
data = "chats"
[settings]
max_seq_length = 2048
block_size = 8
lora_rank = 16
'''),
    _f("postgres-sql", '''
title = "Questions to Postgres SQL"
about = "Plain questions to SQL against your schema, 6,000 pairs"
by = "datawhale-dan"
use = "code"
updated = 2026-11-03
model = "Qwen/Qwen2.5-Coder-14B-Instruct"
data = "qa"
[settings]
max_seq_length = 2048
lora_rank = 32
learning_rate = 1e-4
[measured]
machine = "RTX 4090"
peak_gib = 19.6
s_per_step = 4.8
'''),
    _f("code-review-style", '''
title = "Code review in our team's style"
about = "Review comments the way your seniors write them, from 9,000 merged PRs"
by = "kestrel-dev"
use = "code"
updated = 2026-11-01
model = "Qwen/Qwen3-32B"
data = "chats"
[settings]
max_seq_length = 4096
lora_rank = 16
learning_rate = 1e-4
[measured]
machine = "RTX 5090"
peak_gib = 30.7
s_per_step = 11.2
'''),
    _f("invoices-json", '''
title = "Invoices to JSON"
about = "Vendor, lines, totals and tax out of invoice text, always valid JSON"
by = "ledgerline"
use = "extract"
updated = 2026-11-06
model = "Qwen/Qwen3-4B"
data = "qa"
[settings]
max_seq_length = 2048
lora_rank = 16
learning_rate = 2e-4
train_on = "answers"
[measured]
machine = "RTX 4090"
peak_gib = 9.4
s_per_step = 1.6
'''),
    _f("ticket-triage", '''
title = "Ticket triage, 14 labels"
about = "One label of fourteen per ticket, with the reason in a line"
by = "shopfront-labs"
use = "extract"
updated = 2026-10-24
model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
data = "qa"
[settings]
max_seq_length = 1024
lora_rank = 16
'''),
    _f("less-flattery", '''
title = "Less flattery, straighter answers"
about = "Prefers the answer that disagrees when it should"
by = "evalnerd"
use = "prefer"
updated = 2026-11-04
model = "Qwen/Qwen2.5-7B-Instruct"
data = "pairs"
loop = "dpo"
[settings]
max_seq_length = 1024
learning_rate = 5e-6
dpo_beta = 0.1
[measured]
machine = "RTX 4090"
peak_gib = 18.9
s_per_step = 4.4
'''),
]
