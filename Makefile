saxpy:
	nvcc sample.cu -arch=sm_75 -o sample

debug:
	nvcc -g -G sample.cu -arch=sm_75 --ptxas-options=-v -o sample

clean:
	rm -rf sample
