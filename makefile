lda:lda.cc
	g++ lda.cc -o lda -I /usr/local/include/eigen3/ -std=c++11 -O2 
sparselda:sparselda.cc
	g++ sparselda.cc -o sparselda -I /usr/local/include/eigen3/ -std=c++11 -O2
tot:tot.cc
	g++ tot.cc -o tot -I /usr/local/include/eigen3/ -std=c++11 -O3 
