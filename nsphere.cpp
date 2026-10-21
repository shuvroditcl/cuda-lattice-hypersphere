/* 159.735 Semester 2, Ian Bond, 11/9/2025
 
 Sequential version of the N-sphere counting problem for Assignment
 5. Two alternative algorithms are presented.

 Note: a rethink will be needed when implementing a GPU version of
 this. You can't just cut and paste code.

 To compile: g++ -O3 -o nsphere nsphere.cpp
 (you will get slightly better performance with the O3 optimization flag)

 To run:

 ./nsphere [number of dimensions] [hyper-radius]

 e.g.

 ./nsphere 2 2.05
 ./nsphere 3 1.5
 
*/
#include <cstdlib>
#include <cmath>

#include <iostream>
#include <string>

#include <vector>

long powlong(long n, long k)
/* Evaluate n**k where both are long integers */
{
  long p = 1;
  for (long i = 0; i < k; ++i) p *= n;
  return p;
}

/*----------------------------------------------------------------------------*/

void convert(long num, long base, std::vector<long>& index)
/* Convert a decimal number into another base system - the individual

   digits in the new base are stored in the index array. */
{
  const long ndim = index.size();
  for (long i = 0; i < ndim; ++i) index[i] = 0;
  long idx = 0;
  while (num != 0) {
    long rem = num % base;
    num = num / base;
    index[idx] = rem;
    ++idx;
  }
}

long count_in_v1(long ndim, double radius)
/* 
   Version 1 of the counting algorithm. Given:

   ndim   -> number of dimensions of the hypersphere
   radius -> radius of the hypersphere

   count the number of integer points that lie wholly within the
   hypersphere, assuming it is centred on the origin.
*/
{
  const long halfb = static_cast<long>(floor(radius));
  const long base = 2 * halfb + 1;
  const double rsquare = radius * radius;

  // This is the total number of points we will need to test.
  const long ntotal = powlong(base, ndim);

  long count = 0;

  // Indices in x,y,z,.... 
  std::vector<long> index(ndim, 0);

  // Loop over the total number of points. For each visit of the loop,
  // we covert n to its equivalent in a number system of given "base".
  for (long n = 0; n < ntotal; ++n) {
    convert(n, base, index);
    double rtestsq = 0;
    for (long k = 0; k < ndim; ++k) {
      double xk = index[k] - halfb;
      rtestsq += xk * xk;
    }
    if (rtestsq < rsquare) ++count;
  }

  return count;
}

/*----------------------------------------------------------------------------*/

void addone(std::vector<long>& index, long base, long i)
/* Add one to a digital counter of given base. When one digit hits
   maximum, it is necessary to carry one over into the next
   column. This is done recursively here. */
{
  long ndim = index.size();
  long newv = index[i] + 1;
  if (newv >= base) {
    index[i] = 0;
    if (i < ndim - 1) addone(index, base, i+1);
  }
  else {
    index[i] = newv;
  }
}

long count_in_v2(long ndim, double radius)
/* 
   Version 2 of the counting algorithm. Given:

   ndim   -> number of dimensions of the hypersphere
   radius -> radius of the hypersphere

   count the number of integer points that lie wholly within the
   hypersphere, assuming it is centred on the origin.
*/
{
  const   long halfb   = static_cast<long>(floor(radius));
  const   long base    = 2 * halfb + 1;
  const double rsquare = radius * radius;
  const   long ntotal  = powlong(base, ndim);

  long count = 0;

  // This is the counter
  std::vector<long> index(ndim, 0);

  // Loop over the total number of points to test, ticking over the
  // counter as we go.
  for (long n = 0; n < ntotal; ++n) {
    double rtestsq = 0;
    for (long k = 0; k < ndim; ++k) {
      double xk = index[k] - halfb;
      rtestsq += xk * xk;
    }
    if (rtestsq < rsquare) ++count;
    addone(index, base, 0);
  }
  return count;
}

int main(int argc, char* argv[]) 
{
  const int nd = std::atoi(argv[1]);
  const double r = std::atof(argv[2]);
  const long num1 = count_in_v1(nd, r);
  //const long num2 = count_in_v2(nd, r);
  const long num2 = 0;
  std::cout << " -> " << num1 << " " << num2 << std::endl;

}
