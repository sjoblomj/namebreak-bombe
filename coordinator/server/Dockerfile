FROM rust:1-bookworm AS build
WORKDIR /build
COPY Cargo.toml ./
COPY protocol ./protocol
COPY server ./server
COPY client ./client
RUN cargo build --release -p namebreak-server

FROM debian:bookworm-slim
COPY --from=build /build/target/release/namebreak-server /usr/local/bin/namebreak-server
EXPOSE 8080
ENV BIND_ADDR=0.0.0.0:8080
ENV DATABASE_URL=sqlite:///data/namebreak.db
ENTRYPOINT ["/usr/local/bin/namebreak-server"]
