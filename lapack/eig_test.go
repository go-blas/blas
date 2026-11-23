package lapack_test

import (
	"math"
	"math/big"
	"math/rand/v2"
	"strconv"
	"sync"
	"testing"

	"github.com/go-blas/blas"
	"github.com/go-blas/blas/internal/oracle"
	"github.com/go-blas/blas/lapack"
	"github.com/go-blas/blas/netlib"
)

// eimpl is one implementation of the symmetric eigensolver routines of one
// element type: package lapack, or its exact option netlib.
type eimpl[T float] struct {
	name  string
	syev  func(jobz lapack.Jobz, uplo blas.Uplo, n int, a []T, lda int, w []T) (int, error)
	sytrd func(uplo blas.Uplo, n int, a []T, lda int, d, e, tau []T) error
	orgtr func(uplo blas.Uplo, n int, a []T, lda int, tau []T) error
	steqr func(compz lapack.Compz, n int, d, e, z []T, ldz int) (int, error)
	sterf func(n int, d, e []T) (int, error)
}

var eimpls64 = []eimpl[float64]{
	{"lapack", lapack.Dsyev, lapack.Dsytrd, lapack.Dorgtr, lapack.Dsteqr, lapack.Dsterf},
	{"netlib", netlib.Dsyev, netlib.Dsytrd, netlib.Dorgtr, netlib.Dsteqr, netlib.Dsterf},
}

var eimpls32 = []eimpl[float32]{
	{"lapack", lapack.Ssyev, lapack.Ssytrd, lapack.Sorgtr, lapack.Ssteqr, lapack.Ssterf},
	{"netlib", netlib.Ssyev, netlib.Ssytrd, netlib.Sorgtr, netlib.Ssteqr, netlib.Ssterf},
}

// eigCfg is the block sizes and crossovers of Sytrd, of the QL generation and of
// the QR generation of Orgtr. The small ones make matrices of a few dozen columns
// span many panels, with and without the unblocked tail; the zero one is the
// reference's.
type eigCfg struct{ trdNB, trdNX, qlNB, qlNX, qrNB, qrNX int }

func eigConfigs(name string) []eigCfg {
	if name == "netlib" {
		return []eigCfg{{}} // netlib has the reference's block sizes only
	}
	if testing.Short() {
		return []eigCfg{{3, 2, 3, 2, 3, 2}, {5, 30, 5, 30, 5, 30}, {}}
	}
	// (5, 30): the crossover above the order (n = 25, 26), so that the blocked
	// code is entered with nx = n.
	return []eigCfg{{2, 0, 2, 0, 2, 0}, {3, 2, 3, 2, 3, 2}, {5, 30, 5, 30, 5, 30}, {}}
}

func setEig(t *testing.T, c eigCfg) {
	t.Helper()
	if c != (eigCfg{}) {
		lapack.SetEig(t, c.trdNB, c.trdNX, c.qlNB, c.qlNX, c.qrNB, c.qrNX)
	}
}

func eigName(c eigCfg) string {
	if c == (eigCfg{}) {
		return ""
	}
	return "/nb" + strconv.Itoa(c.trdNB) + "nx" + strconv.Itoa(c.trdNX)
}

// curName and curCfg are the implementation and configuration of the running
// subtest (the tests do not run in parallel).
var (
	curName string
	curCfg  eigCfg
)

// skipSize reports whether the running implementation and configuration is not
// tested at order n, the exact arithmetic of the oracles being what costs: the
// small block sizes (whose point is many panels on few columns) up to order 33,
// (3, 2) up to 65, netlib up to 65, the default configuration of lapack at every
// size.
func skipSize(n int) bool {
	switch {
	case curName == "netlib":
		return n > 65
	case curCfg == eigCfg{}:
		return false
	case curCfg.trdNB == 3:
		return n > 65
	}
	return n > 33
}

func eachE(t *testing.T, f64 func(t *testing.T, im eimpl[float64]), f32 func(t *testing.T, im eimpl[float32])) {
	t.Helper()
	for _, im := range eimpls64 {
		for _, c := range eigConfigs(im.name) {
			t.Run("float64/"+im.name+eigName(c), func(t *testing.T) {
				defer noPanic(t)
				curName, curCfg = im.name, c
				setEig(t, c)
				f64(t, im)
			})
		}
	}
	for _, im := range eimpls32 {
		for _, c := range eigConfigs(im.name) {
			t.Run("float32/"+im.name+eigName(c), func(t *testing.T) {
				defer noPanic(t)
				curName, curCfg = im.name, c
				setEig(t, c)
				f32(t, im)
			})
		}
	}
}

// ---- the bounds ----
//
// The constants k of gamma_k are the ones of the package comment: tested, not
// proved, each at least 4 times the worst ratio value/(gamma_k * norm) measured
// over this corpus (the calibration table of docs/design.md, decision 12, and the
// log of -v, which prints the running worst ratios per bound and precision, both
// with the constant the bound is stated with).
func kSytrd(n int) int { return 16*n + 32 }
func kSteqr(n int) int { return 30 * n }
func kSyev(n int) int  { return 16 * n }

var (
	calMu sync.Mutex
	calib = map[string]float64{}
)

// record keeps the worst ratio of a bound, and fails the test when it exceeds 1.
func record(t *testing.T, what string, ratio float64) {
	t.Helper()
	calMu.Lock()
	if ratio > calib[what] || math.IsNaN(ratio) {
		calib[what] = ratio
	}
	calMu.Unlock()
	if !(ratio <= 1) {
		t.Fatalf("%s: value/bound = %g > 1", what, ratio)
	}
}

func reportCalib(t *testing.T) {
	calMu.Lock()
	defer calMu.Unlock()
	for k, v := range calib {
		t.Logf("calibration %-40s worst value/bound = %.4g", k, v)
	}
}

func tag[T float](what string) string {
	var z T
	if _, ok := any(z).(float32); ok {
		return what + "/float32"
	}
	return what + "/float64"
}

func denorm[T float]() float64 {
	var z T
	if _, ok := any(z).(float32); ok {
		return math.Ldexp(1, -149)
	}
	return math.Ldexp(1, -1074)
}

// ratio is v / (gamma_k * norm + slack), with slack = k*denorm: the data a few
// orders above the subnormals has no relative accuracy below its own granularity.
func ratio[T float](v, norm *big.Float, k int) float64 {
	slack := bf().Mul(bf().SetInt64(int64(k)), bf().SetFloat64(denorm[T]()))
	lim := bf().Add(bf().Mul(oracle.Gamma[T](k), norm), slack)
	if lim.Sign() == 0 {
		if v.Sign() == 0 {
			return 0
		}
		return math.Inf(1)
	}
	r, _ := bf().Quo(v, lim).Float64()
	return r
}

// scales are the factors the data are multiplied by: 1, above RMAX (so that the
// scaling of Syev happens), below RMIN, and the neighbourhood of the subnormals.
func scales[T float]() []float64 {
	var z T
	if _, ok := any(z).(float32); ok {
		return []float64{1, 1e20, 1e-20, 1e-30}
	}
	return []float64{1, 1e150, 1e-150, 1e-300}
}

// ---- matrices ----

// eigSizes are the orders: around the block sizes 32 and the crossovers, and one
// beyond 256.
func eigSizes() []int {
	if testing.Short() {
		// The emulated lanes (qemu, -short) ran past go test's 10 minutes with
		// the exact oracles at 63 to 65, and Syev alone took 70 s natively at
		// 24 to 33.
		return []int{0, 1, 2, 3, 25, 33}
	}
	// 129 spans four panels of the default block size 32; the -race lanes
	// (Windows above all) ran past 30 minutes with 127, 128 and 257 as well.
	return []int{0, 1, 2, 3, 24, 25, 26, 31, 32, 33, 65, 129}
}

// kindsFor returns the matrix families a size is tested on: all of them for the
// small ones, fewer where the exact arithmetic of the oracles costs most.
func kindsFor(n int) []string {
	all := []string{}
	for _, k := range spectraKinds {
		all = append(all, "dense/"+k)
	}
	for _, k := range tridKinds {
		all = append(all, "trid/"+k)
	}
	switch {
	case n <= 33:
		return all
	case n <= 65:
		return []string{"dense/random", "dense/clustered", "trid/wilkinson", "trid/clement"}
	case n <= 129:
		return []string{"dense/random", "trid/tinyoff"}
	}
	return []string{"dense/random"}
}

// denseSym returns a column-major symmetric n x n matrix of a family: the triangle
// up names holds it (multiplied by scale), the other triangle holds NaN (it must
// not be read), the padding of lda the sentinel.
func denseSym[T float](kind string, n, lda int, up bool, scale float64, rng *rand.Rand) []T {
	full := make([]float64, n*n)
	switch {
	case len(kind) > 6 && kind[:6] == "dense/":
		full = spread(spectrum(kind[6:], n, rng), rng)
	default:
		d, e := tridiag(kind[5:], n, rng)
		for i := 0; i < n; i++ {
			full[i+i*n] = d[i]
			if i < n-1 {
				full[i+1+i*n], full[i+(i+1)*n] = e[i], e[i]
			}
		}
	}
	a := make([]T, matLen(n, n, lda))
	for p := range a {
		a[p] = sentinel
	}
	for j := 0; j < n; j++ {
		for i := 0; i < n; i++ {
			switch {
			case i == j || (up && i < j) || (!up && i > j):
				a[i+j*lda] = T(full[i+j*n] * scale)
			default:
				a[i+j*lda] = nan[T]()
			}
		}
	}
	return a
}

// sameOutside asserts that the elements of a outside the n x n leading block (the
// padding) and, when other is set, in the triangle opposite to up are as in a0.
func sameOutside[T float](t *testing.T, what string, a, a0 []T, n, lda int, up, other bool) {
	t.Helper()
	for j := 0; j < n; j++ {
		for i := 0; i < lda && i+j*lda < len(a); i++ {
			outside := i >= n
			opp := i < n && ((up && i > j) || (!up && i < j))
			if (outside || (other && opp)) && !same(a[i+j*lda:i+j*lda+1], a0[i+j*lda:i+j*lda+1]) {
				t.Fatalf("%s: element (%d,%d) outside what the routine may write was modified", what, i, j)
			}
		}
	}
}

func triName(up bool) string {
	if up {
		return "U"
	}
	return "L"
}

func uploOf(up bool) blas.Uplo {
	if up {
		return blas.Upper
	}
	return blas.Lower
}

// ---- Sytrd and Orgtr ----

func TestSytrdOrgtr(t *testing.T) {
	defer reportCalib(t)
	eachE(t, sytrdOrgtrT[float64], sytrdOrgtrT[float32])
}

func sytrdOrgtrT[T float](t *testing.T, im eimpl[T]) {
	rng := newRNG(11)
	for _, n := range eigSizes() {
		if skipSize(n) || n > 129 {
			continue
		}
		for _, kind := range kindsFor(n) {
			for _, up := range []bool{true, false} {
				lda := max(1, n) + int(rng.IntN(3))
				a0 := denseSym[T](kind, n, lda, up, 1, rng)
				a := clone(a0)
				d, e, tau := make([]T, n), make([]T, max(n-1, 0)), make([]T, max(n-1, 0))
				what := kind + "/n" + strconv.Itoa(n) + "/" + triName(up)
				if err := im.sytrd(uploOf(up), n, a, lda, d, e, tau); err != nil {
					t.Fatalf("%s: sytrd: %v", what, err)
				}
				sameOutside(t, what+" sytrd", a, a0, n, lda, up, true)
				q := clone(a)
				if err := im.orgtr(uploOf(up), n, q, lda, tau); err != nil {
					t.Fatalf("%s: orgtr: %v", what, err)
				}
				sameOutside(t, what+" orgtr", q, a, n, lda, up, false)
				if n == 0 {
					continue
				}
				k := kSytrd(n)
				qb := toBM(q, n, n, lda)
				tb := tridBM(d, e, n)
				a0b := symFull(a0, n, lda, up)
				res := sub(a0b, mul(mul(qb, tb), qb.T()))
				record(t, tag[T]("sytrd A-QTQ'"), ratio[T](fro(res), fro(a0b), k))
				sq := bf().Sqrt(bf().SetInt64(int64(n)))
				record(t, tag[T]("orgtr Q'Q-I"), ratio[T](orthDefect(q, n, lda), sq, k))
				if n <= 33 {
					// Steqr with COMPZ = 'V': the eigenvectors of A itself.
					dd, ee, z := clone(d), clone(e), clone(q)
					if info, err := im.steqr(lapack.CompzOriginal, n, dd, ee, z, lda); err != nil || info != 0 {
						t.Fatalf("%s: steqr V: info %d, err %v", what, info, err)
					}
					res := sub(mul(a0b, toBM(z, n, n, lda)), scaleCols(z, n, lda, dd))
					record(t, tag[T]("sytrd+steqr V AZ-ZL"), ratio[T](fro(res), fro(a0b), kSyev(n)))
				}
			}
		}
	}
}

// ---- Steqr and Sterf ----

func TestSteqr(t *testing.T) {
	defer reportCalib(t)
	eachE(t, steqrT[float64], steqrT[float32])
}

func steqrT[T float](t *testing.T, im eimpl[T]) {
	rng := newRNG(12)
	for _, n := range eigSizes() {
		if skipSize(n) || n > 129 {
			continue
		}
		for ki, kind := range tridKinds {
			if n > 33 && kind != "random" && kind != "wilkinson" && kind != "tinyoff" {
				continue
			}
			if n > 129 && ki != 0 {
				continue
			}
			sc := scales[T]()[(n+ki)%4]
			d64, e64 := tridiag(kind, n, rng)
			d0, e0 := make([]T, n), make([]T, max(n-1, 0))
			for i := range d0 {
				d0[i] = T(d64[i] * sc)
			}
			for i := range e0 {
				e0[i] = T(e64[i] * sc)
			}
			what := kind + "/n" + strconv.Itoa(n) + "/s" + strconv.FormatFloat(sc, 'g', 3, 64)
			var ev []*big.Float
			if n > 0 && n <= 129 {
				ev = sturmEigs(d0, e0, n)
			}
			// Eigenvalues only: Steqr and Sterf.
			for name, run := range map[string]func(d, e []T) (int, error){
				"steqr N": func(d, e []T) (int, error) { return im.steqr(lapack.CompzNone, n, d, e, nil, 1) },
				"sterf":   func(d, e []T) (int, error) { return im.sterf(n, d, e) },
			} {
				d, e := clone(d0), clone(e0)
				if info, err := run(d, e); err != nil || info != 0 {
					t.Fatalf("%s %s: info %d, err %v", name, what, info, err)
				}
				checkEigs[T](t, name+" "+what, d, ev, n, kSteqr(n))
			}
			// Eigenvectors of the tridiagonal matrix.
			d, e := clone(d0), clone(e0)
			ldz := max(1, n) + rng.IntN(2)
			z := make([]T, matLen(n, n, ldz))
			for p := range z {
				z[p] = sentinel
			}
			info, err := im.steqr(lapack.CompzIdentity, n, d, e, z, ldz)
			if err != nil || info != 0 {
				t.Fatalf("steqr I %s: info %d, err %v", what, info, err)
			}
			checkEigs[T](t, "steqr I "+what, d, ev, n, kSteqr(n))
			if n == 0 {
				continue
			}
			tb := tridBM(d0, e0, n)
			res := sub(mul(tb, toBM(z, n, n, ldz)), scaleCols(z, n, ldz, d))
			// ||T||_F is what the residual is relative to.
			record(t, tag[T]("steqr TZ-ZL"), ratio[T](fro(res), fro(tb), kSteqr(n)))
			record(t, tag[T]("steqr Z'Z-I"), ratio[T](orthDefect(z, n, ldz), bf().Sqrt(bf().SetInt64(int64(n))), kSteqr(n)))
			for j := 0; j < n; j++ {
				for i := n; i < ldz && i+j*ldz < len(z); i++ {
					if z[i+j*ldz] != sentinel {
						t.Fatalf("steqr I %s: padding of z modified", what)
					}
				}
			}
		}
	}
}

// checkEigs asserts that d is ascending and within gamma_k ||T||_2 of the exact
// eigenvalues ev (when there are some), ||T||_2 being the largest of them.
func checkEigs[T float](t *testing.T, what string, d []T, ev []*big.Float, n, k int) {
	t.Helper()
	for i := 1; i < n; i++ {
		if d[i] < d[i-1] {
			t.Fatalf("%s: eigenvalues not ascending at %d: %g > %g", what, i, d[i-1], d[i])
		}
	}
	if ev == nil || n == 0 {
		return
	}
	nrm := bf().Abs(ev[0])
	if x := bf().Abs(ev[n-1]); x.Cmp(nrm) > 0 {
		nrm = x
	}
	worst := 0.0
	for i := 0; i < n; i++ {
		diff := bf().Abs(bf().Sub(oracle.B(d[i]), ev[i]))
		if r := ratio[T](diff, nrm, k); r > worst {
			worst = r
		}
	}
	record(t, tag[T]("eigenvalues vs O-sturm "+what[:5]), worst)
}

// ---- Syev ----

func TestSyev(t *testing.T) {
	defer reportCalib(t)
	eachE(t, syevT[float64], syevT[float32])
}

func syevT[T float](t *testing.T, im eimpl[T]) {
	rng := newRNG(13)
	sc := scales[T]()
	idx := 0
	for _, n := range eigSizes() {
		if skipSize(n) {
			continue
		}
		for _, kind := range kindsFor(n) {
			for _, up := range []bool{true, false} {
				idx++
				s := sc[idx%4]
				lda := max(1, n) + rng.IntN(3)
				a0 := denseSym[T](kind, n, lda, up, s, rng)
				what := kind + "/n" + strconv.Itoa(n) + "/" + triName(up) + "/s" + strconv.FormatFloat(s, 'g', 3, 64)
				var exact []*big.Float
				switch {
				case n == 0:
				case kind[:5] == "trid/" && n <= 129:
					d, e := make([]T, n), make([]T, n-1)
					for i := range d {
						d[i] = a0[i+i*lda]
					}
					for i := range e {
						if up {
							e[i] = a0[i+(i+1)*lda]
						} else {
							e[i] = a0[i+1+i*lda]
						}
					}
					exact = sturmEigs(d, e, n)
				case n <= 33 || (n == 64 && idx%3 == 0):
					exact = jacobiEigs(symFull(a0, n, lda, up))
				}
				// Eigenvectors.
				a := clone(a0)
				w := make([]T, n)
				info, err := im.syev(lapack.EigenvaluesAndVectors, uploOf(up), n, a, lda, w)
				if err != nil || info != 0 {
					t.Fatalf("syev V %s: info %d, err %v", what, info, err)
				}
				for i := 1; i < n; i++ {
					if w[i] < w[i-1] {
						t.Fatalf("syev V %s: not ascending at %d", what, i)
					}
				}
				for j := 0; j < n; j++ {
					for i := n; i < lda && i+j*lda < len(a); i++ {
						if a[i+j*lda] != sentinel {
							t.Fatalf("syev V %s: padding modified", what)
						}
					}
				}
				if n > 0 {
					ab := symFull(a0, n, lda, up)
					norm := fro(ab)
					res := sub(mul(ab, toBM(a, n, n, lda)), scaleCols(a, n, lda, w))
					record(t, tag[T]("syev AZ-ZL"), ratio[T](fro(res), norm, kSyev(n)))
					// O-res, per column.
					worst := 0.0
					for j := 0; j < n; j++ {
						if r := ratio[T](colNorm2(res, j), norm, kSyev(n)); r > worst {
							worst = r
						}
					}
					record(t, tag[T]("syev column residual"), worst)
					record(t, tag[T]("syev Z'Z-I"), ratio[T](orthDefect(a, n, lda), bf().Sqrt(bf().SetInt64(int64(n))), kSyev(n)))
					if exact != nil {
						worst := 0.0
						for i := 0; i < n; i++ {
							diff := bf().Abs(bf().Sub(oracle.B(w[i]), exact[i]))
							if r := ratio[T](diff, norm, kSyev(n)); r > worst {
								worst = r
							}
						}
						record(t, tag[T]("syev eigenvalues vs exact"), worst)
					}
				}
				// Eigenvalues only: the triangle is destroyed, nothing else changes.
				a = clone(a0)
				w2 := make([]T, n)
				info, err = im.syev(lapack.EigenvaluesOnly, uploOf(up), n, a, lda, w2)
				if err != nil || info != 0 {
					t.Fatalf("syev N %s: info %d, err %v", what, info, err)
				}
				sameOutside(t, "syev N "+what, a, a0, n, lda, up, true)
				if exact != nil {
					norm := fro(symFull(a0, n, lda, up))
					worst := 0.0
					for i := 0; i < n; i++ {
						diff := bf().Abs(bf().Sub(oracle.B(w2[i]), exact[i]))
						if r := ratio[T](diff, norm, kSyev(n)); r > worst {
							worst = r
						}
					}
					record(t, tag[T]("syev N eigenvalues vs exact"), worst)
				}
			}
		}
	}
}
