#include <NTL/ZZ_pXFactoring.h>
#include <NTL/ZZ_pEX.h>

NTL_CLIENT

int main()
{
  ZZ_p::init(to_ZZ(17)); // define GF(17)

  ZZ_pX P;
  BuildIrred(P, 10); // generate an irreducible polynomial P
  // of degree 10 over GF(17)

  ZZ_pE::init(P); // define GF(17^10)

  ZZ_pEX f, g, h;  // declare polynomials over GF(17^10)

  random(f, 20);  // f is a random, monic polynomial of degree 20
  SetCoeff(f, 20);

  random(h, 20); // h is a random polynomial of degree less than 20

  g = MinPolyMod(h, f); // compute the minimum polynomial of h modulo f

  if (g == 0) Error("oops (1)"); // check that g != 0

  if (CompMod(g, h, f) != 0) // check that g(h) = 0 mod f
    Error("oops (2)");
}
