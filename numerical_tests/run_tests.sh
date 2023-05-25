# first test
mkdir -p test_reconstruction
for noise in {0,0.2,0.4,0.8}
do
    python reconstruction.py $noise
    mkdir -p test_reconstruction/reconstruction_${noise}
    mv *.pdf *.csv test_reconstruction/reconstruction_${noise}
done
python classifier.py