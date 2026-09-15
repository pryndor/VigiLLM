# Team Workflow

## The four systems and what each member touches

| System | Who touches it | For what |
|---|---|---|
| GitHub repo | Any member, anytime | Code, training scripts, rule engine, configs — normal PR flow |
| Kaggle/Colab | Whoever's turn it is to train | Burst GPU compute, ~9hr sessions |
| HuggingFace Hub | Whoever just finished a training turn | Push updated checkpoint/dataset |
| Weights & Biases | Automatic during training | Shared experiment dashboard, everyone can view |

## Actual accounts/repos (2026-09-15)

- **GitHub**: github.com/pryndor/VigiLLM (private)
- **HuggingFace**: personal namespace `Pryndor`, not an org — `Pryndor/vigillm-adapter` (checkpoints) and `Pryndor/vigillm-dataset` (training data), both private. Skipped creating an HF org since it's browser-only (no CLI path) and adds friction without real benefit for a small team — HF supports adding collaborators directly to individual repos, which covers the same need. Team members get added as repo-level collaborators (repo Settings → Collaborators) rather than org members.
- **W&B**: project at wandb.ai/balathepharmacist-drugvigil/vigillm

## Compute pooling

Each team member has their own free Kaggle account (30hr/week GPU quota each) and free Colab account (overflow). A 4-person team has 120hr/week pooled Kaggle time without spending anything — just coordinate whose turn it is.

## Avoiding checkpoint collisions

Two people training the same shared checkpoint simultaneously causes a race condition on push (whoever pushes second overwrites the first's work, or HF Hub reports a conflict). Solution: **round-robin turns**.

- Shared tracking board (a GitHub Issue titled "Training Queue", or a shared spreadsheet) — one row per turn: member name, start time, checkpoint version pulled, checkpoint version pushed, eval metrics after this turn
- Claim a turn by commenting/editing the board before starting a Kaggle session
- Only start training after confirming no one else's turn is active

This is deliberately simple (no automated locking system) — a 4-person team doesn't need infrastructure for this, just a shared doc and the discipline to check it.

## Parallel work that DOESN'T need to wait for a turn

- Code changes (training script, rule engine, data schema) — normal GitHub branches/PRs, doesn't touch the shared checkpoint
- Synthetic data generation (distillation queries to teacher models) — writes to `data/`, separate from the checkpoint lineage, can run in parallel with someone else's training turn
- Evaluation runs against a *downloaded copy* of the latest checkpoint — read-only, doesn't push anything, no conflict risk

Only the actual "resume training and push a new checkpoint" step needs the round-robin discipline.

## Advanced option (once round-robin becomes a bottleneck)

If turn-waiting becomes a real constraint as the team grows: switch to **parallel LoRA adapters**. Each member trains their own adapter branch on their own data shard/session independently (no shared checkpoint to collide over), and adapters get merged periodically (weighted average, since they share the same base architecture) by whoever's turn it is to do the merge. More complex to set up — not needed for a small team starting out, documented here for later.

## Code review discipline

Same as any software project: PRs for training script changes, rule engine changes, or schema changes get reviewed by at least one other team member before merging into `main`. Training runs should reference a specific commit hash of the training script (log it in W&B run config) so any result is reproducible back to the exact code that produced it.

## Communication

Shared tracking board doubles as informal standup — checking it tells you what happened during turns you weren't present for (whose turn, what changed, eval metrics trend) without needing a separate status meeting.
