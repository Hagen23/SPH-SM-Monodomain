.SUFFIXES: .cpp .cu

CC = nvcc
CFLAGS = -std=c++11 -O3 -Xcompiler -ansi -Xcompiler -Ofast -Wno-deprecated-gpu-targets
MATH3D = Math3D/
INCLUDES = -I Math3D/ -I SPH_SM_monodomain/ -I $CUDA_HOME/include/ -I cuda_common/
LDFLAGS = -lGL -lglut -lGLU
DEBUGF = $(CFLAGS) -ggdb
SOURCES = *.cpp Math3D/*.cpp SPH_SM_monodomain/*.cpp
OUTF = build/
MKDIR_P = mkdir -p

all: build

directory: $(OUTF)

build: directory $(SOURCES) $(OUTF)sph_sm_m

$(OUTF):
	$(MKDIR_P) $(OUTF)

$(OUTF)sph_sm_m: $(OUTF)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(OUTF)sph_sm $(SOURCES) $(LDFLAGS)

debug: directory $(SOURCES) $(OUTF)sph_sm_m_d

$(OUTF)sph_sm_m_d: $(OUTF)
	$(CC) $(DEBUGF) $(INCLUDES) -o $(OUTF)sph_sm_d $(SOURCES) $(LDFLAGS) 

clean:
	@[ -f $(OUTF)sph_sm_m ] && rm $(OUTF)sph_sm || true
	@[ -f $(OUTF)sph_sm_m_d ] && rm $(OUTF)sph_sm_d || true

rebuild: clean build

redebug: clean debug