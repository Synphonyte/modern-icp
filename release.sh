#!/bin/bash

cargo +nightly fmt --check &&
cargo +nightly clippy --tests -- -D warnings &&
cargo +nightly test --features modelz &&
cargo +nightly rdme --check &&
retag_and_push
