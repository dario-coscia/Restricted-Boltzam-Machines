# first test
mkdir -p test_reconstruction
for noise in {0.1,0.5}; #{0,0.1,0.5}
do
    python reconstruction.py $noise
    mkdir -p test_reconstruction/reconstruction_${noise}
    mv *.pdf *.csv test_reconstruction/reconstruction_${noise}
done
