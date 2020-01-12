cd ./knn_rust &&\
cargo +nightly build --release --all-features &&\
cd .. &&\
cp "./knn_rust/target/release/libknn_rust.so" ./knn_python/knn_rust.so
