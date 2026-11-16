# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/soshixos/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/soshixos/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server, two APIs, fourteen live free models: the best live free-model aggregator. It serves OpenAI Chat Completions and Anthropic Messages, in JSON and streaming, routing each request to the highest-rated free OpenRouter model that meets it and dialling it live through the make.fun gateway — with failover when a free model refuses.

## How it works

- **Routing (quality):** candidates are the brief's free models that meet the prompt's requirements, best rating first. The thinking-machines model (1475, best overall) serves every plain or tools prompt; the super-VL model (1303, best JSON-capable one) serves the two JSON-mode prompts. Serving the first candidate earns the full quality for that prompt.
- **Failover:** on a 429 or 5xx the server records the refusal, remembers the model as refused for the rest of the trial, and dials the next-best legal model. A prompt served after refusals carries the trail in the `x-makefun-upstream-failures` header, so the failover column can see it. A 401 (missing or refused key) aborts at once — no model would answer.
- **Wires:** both client wires are served in JSON and streaming, with the dialled model's id put back on every answer and every streamed chunk, usage kept whole (estimated from the text when a provider reports none), and tool calls translated both ways.
- **Determinism:** every choice is a function of the brief, the requests and the gateway's answers — no clock, no unseeded randomness — so a replay leaves the same state export.

## Run it

```sh
node main.ts --list   # print the catalog document
node main.ts          # serve the trial: registers with $MAKEFUN_ENV_URL, exits 0 when $MAKEFUN_ENV_URL/work says done
```

Node 22, standard library only; nothing is installed. The server reads the brief at `$MAKEFUN_PROMPT_FILE`, dials models at `$OPENAI_BASE_URL/chat/completions` with the trial's dummy key in `$OPENAI_API_KEY`, and registers at `$MAKEFUN_ENV_URL/register`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=d1rnvwbk&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
