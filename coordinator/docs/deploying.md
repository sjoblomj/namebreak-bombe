# Configuring and deploying the server

The server is configured with environment variables, all but `ADMIN_TOKEN`
optional.

## Server configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite://namebreak.db` | SQLite connection string |
| `ADMIN_TOKEN` | *(required)* | protects `/admin/*` |
| `BIND_ADDR` | `0.0.0.0:8080` | listen address |
| `TARGET_CHUNK_SECONDS` | `900` | desired wall-clock time per range |
| `DEFAULT_RATE_PER_SEC` | `500000000` | assumed candidates/sec until a user's first completed range refines it |
| `DEFAULT_DICTIONARY_RATE_PER_SEC` | `1000000000` | the same for a dictionary target's ranges, measured apart |
| `DEFAULT_MEET_IN_THE_MIDDLE_RATE_PER_SEC` | `1000000000` | the same for a meet-in-the-middle target's ranges, in tails a second (a GPU rewinds about 2 billion, a CPU 0.07) |
| `MIN_CHUNK_CANDIDATES` / `MAX_CHUNK_CANDIDATES` | `1000000` / `1000000000000000000` | clamp on carved chunk size |
| `LEASE_SECONDS` | `21600` (6 hours) | how long a claimed range stays leased after the last sign of life from its client (the claim, then each heartbeat) |
| `STALL_RELEASE_SECONDS` | `86400` (a day) | how long a range stays leased while its heartbeats report no new progress (a paused or stuck client) |
| `RECLAIM_INTERVAL_SECS` | `30` | how often expired leases are swept back to pending |
| `EMA_ALPHA` | `0.3` | smoothing factor for each user's observed-rate average |
| `CANARY_PROBABILITY` | `0.33` | chance that a claim gets a canary instead of real work; `0` turns them off |
| `CANARY_SECONDS` | `5` | how long a canary should take, at the client's measured rate |
| `DICTIONARY_PATH` | *(see meaning)* | the English word list automatic priority ranges rank prefixes by; unset, the first of `/usr/share/dict/words`, `american-english` and `british-english` there is |
| `MIN_FREE_DISK_BYTES` | `268435456` (256 MiB) | below this much free on the database's disk, reported basenames are discarded |
| `WARN_FREE_DISK_BYTES` | `536870912` (512 MiB) | below this much free, the dashboard warns that basenames soon won't be kept |

## Deploying to fly.io

```sh
cd coordinator
fly launch --no-deploy   # picks up fly.toml; say no to Postgres/Redis add-ons
fly volumes create namebreak_data --size 1   # matches the [[mounts]] in fly.toml
fly secrets set ADMIN_TOKEN=<a real secret>
fly deploy
```

The client is not part of the server image - volunteers build `namebreak`
(`../client`) locally and run its `coordinator` mode with
`server_url = https://<your-app>.fly.dev` in their `config.conf`.
