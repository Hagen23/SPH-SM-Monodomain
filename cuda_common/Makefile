
.SUFFIXES: .cpp .cu

CC = nvcc
CFLAGS = -std=c++11 -O3 -Xcompiler -ansi -Xcompiler -Ofast -Wno-deprecated-gpu-targets
INCLUDES = -I $CUDA_HOME/include/ -I ../cuda_common/
LDFLAGS = -lGL -lglut -lGLU -lGLEW
DEBUGF = $(CFLAGS) -ggdb

SOURCES = main.cpp 
SOURCES_CU = simpleKernelVBO.cu

OBJECTS=$(SOURCES:.cpp=.o)
OBJECTSCU=$(SOURCES_CU:.cu=.o)

OUTDIR = build/
ODIR = .o_dir/
OUTFILE = cuda3d_vis
OUTDEBUG = cuda3d_vis_d
MKDIR_P = mkdir -p

all: build

directory: $(OUTDIR) $(ODIR)

build: directory $(SOURCES) $(OUTDIR)$(OUTFILE)

$(OUTDIR):
	$(MKDIR_P) $(OUTDIR)

$(ODIR):
	$(MKDIR_P) $(ODIR)

$(OUTDIR)$(OUTFILE) : $(OUTDIR) $(OBJECTS) $(OBJECTSCU)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(ODIR)$(OBJECTS) $(ODIR)$(OBJECTSCU) $(LDFLAGS)

debug: directory $(SOURCES) $(OUTDIR)$(OUTDEBUG)

$(OUTDIR)$(OUTDEBUG) : $(OUTDIR) $(OBJECTS) $(OBJECTSCU)
	$(CC) $(DEBUGF) $(INCLUDES) -o $@ $(ODIR)$(OBJECTS) $(ODIR)$(OBJECTSCU) $(LDFLAGS) 

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) $< -c $@
	mv $@ $(ODIR)

.cu.o:
	$(CC) $(CFLAGS) $(INCLUDES) $< -c $@
	mv $@ $(ODIR)

clean:
	@[ -f $(OUTDIR)$(OUTFILE) ] && rm $(OUTDIR)$(OUTFILE) || true
	@[ -f $(OUTDIR)$(OUTDEBUG)  ] && rm $(OUTDIR)$(OUTDEBUG)  || true
	rm $(ODIR)*.o

rebuild: clean build

redebug: clean debug