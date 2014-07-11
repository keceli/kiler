/* Defines the SIPS data structure. */

#include "slepceps.h"

typedef struct {
  MPI_Comm     world,mat,eps;
  PetscMPIInt  id,idMat,idEps,nproc,nprocMat,nprocEps;
} EIG_Comm; /* communicators, number of processors and ids */

PETSC_EXTERN PetscErrorCode EigCommCreate(MPI_Comm,PetscMPIInt,EIG_Comm**);
PETSC_EXTERN PetscErrorCode EigCommDestroy(EIG_Comm*);
PETSC_EXTERN PetscErrorCode EigCommGetMatComm(EIG_Comm *,MPI_Comm *);
PETSC_EXTERN PetscErrorCode EigCommGetEpsComm(EIG_Comm *,MPI_Comm *);
PETSC_EXTERN PetscErrorCode EigCommGetEpsId(EIG_Comm *commEig,PetscMPIInt*);
PETSC_EXTERN PetscErrorCode SetCommEig(MPI_Comm,Mat,EIG_Comm*);
PETSC_EXTERN PetscErrorCode SetCommMat(MPI_Comm,int,MPI_Comm[],int*);
PETSC_EXTERN PetscErrorCode SetCommEps(MPI_Comm,int,MPI_Comm[],int*);
PETSC_EXTERN PetscErrorCode EPSGetMatInertia(EPS,PetscReal,PetscInt*);
PETSC_EXTERN PetscErrorCode NevalsGetExtremeValues(PetscInt*,PetscInt,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSGetLocalInterval(EPS,EIG_Comm*,PetscReal,PetscReal,PetscReal*,PetscReal*);

#if defined(MV)
extern PetscErrorCode SetLocalEigRange(EIG_Comm*,EVBD*,SIGMA*,PetscInt,EVBD*,PetscBool*); 
extern PetscErrorCode SetInitialSigma(EIG_Comm*,PetscInt,PetscInt,EVBD*,PetscInt*,SIGMA*);
extern PetscErrorCode GetClusterEig(SIGMA*,EVBD*,PetscReal,EVSOL*);

extern PetscErrorCode DumpEndClusters(int,int,int*,int*,PetscScalar*,int*,int*,SIGMA*,int[]);
extern PetscErrorCode CkBookEvs(EPS,SIGMA*,EVBD*,EVSOL*,EVSOL*,PetscReal,int,int,int);
extern PetscErrorCode AddEvsol(EPS,int,int,int,SIGMA*,EVBD*,EVSOL*,EVSOL*,PetscBool*);
extern PetscErrorCode SetNewShifts(PetscInt,EVSOL*,EVSOL*,EVBD *,EVBD *,
                       PetscInt,PetscInt,PetscInt,PetscInt*,SIGMA*);
extern PetscErrorCode GetNextShift(EVSOL*,EVBD*,PetscInt*,PetscInt*,PetscInt*,SIGMA*);
extern PetscErrorCode SendDoneToNb(EIG_Comm *,NBCOMM*,EVBD*,int); 
extern PetscErrorCode UpdateEvdone(int,int,EVBD*,EVSOL*);
extern PetscErrorCode RecvFromNb(EIG_Comm*,NBCOMM*,EVBD*,EVSOL*,int,SIGMAS*,SIGMA*);
extern PetscErrorCode RecvDoneFromNb(EIG_Comm *,NBCOMM*,EVBD*,int);
extern PetscErrorCode GetParentShift(SIGMA *,EVSOL *,EVBD *,PetscInt *);
extern PetscErrorCode FinalCheckEvs(EVBD *,EVSOL *,PetscInt *,SIGMA *,PetscMPIInt,PetscMPIInt,PetscInt);
extern PetscErrorCode GetGlobalEval(SIPS,PetscInt*,PetscReal*);
extern PetscErrorCode DispTime(SIPS,Mat,PetscLogDouble,PetscInt,PetscInt,PetscInt);

#endif
