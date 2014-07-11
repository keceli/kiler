

CFLAGS     =
FFLAGS     =
CPPFLAGS   =
FPPFLAGS   =
DIRS       = 
LOCDIR     = 
EXAMPLESC  = matview.c 
EXAMPLESF  = 
MANSEC     = 

include ${SLEPC_DIR}/conf/slepc_common

matview: matview.o  chkopts
	-${CLINKER} -o matview matview.o  ${SLEPC_LIB}
	${RM} matview.o 
