FROM rust:1-bookworm AS build
WORKDIR /build
COPY Cargo.toml ./
COPY protocol ./protocol
COPY server ./server
RUN cargo build --release -p namebreak-server

FROM debian:bookworm-slim
# /usr/share/dict/words, for automatic priority ranges - see src/likely_prefixes.rs.
RUN apt-get update && apt-get install -y --no-install-recommends wamerican && rm -rf /var/lib/apt/lists/*
COPY --from=build /build/target/release/namebreak-server /usr/local/bin/namebreak-server
EXPOSE 8080
ENV BIND_ADDR=0.0.0.0:8080
ENV DATABASE_URL=sqlite:///data/namebreak.db
ENTRYPOINT ["/usr/local/bin/namebreak-server"]
