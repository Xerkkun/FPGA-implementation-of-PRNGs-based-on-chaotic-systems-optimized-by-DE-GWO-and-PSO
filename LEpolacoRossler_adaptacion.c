#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
Para el sistema de Lorenz las ecuaciones son
\dot{x} = sigma(y-x)
\dot{y} = x(rho-z) - y
\dot{z} = xy - beta z

Su Jacobiana es

J[0] = sigma( y' - x' )
J[1] = (rho - z)x' - y' - x z'
J[2] = yx' + xy' - beta z'

 Esta función se debe de cambiar para cualquier
otro oscilador
**/
void sistema(double *out, double *in, double *par)
{
    double x, y, z;

    x = in[0];
    y = in[1];
    z = in[2];
    out[0] = -y - z;
    out[1] = x + par[0]*y;
    out[2] = par[1] + z*(x - par[2]);
}

void jacobiana(double *J, double *in, double *inp, double *par)
{
    J[0] = -inp[1] - inp[2];
    J[1] = inp[0] + par[0]*inp[1] ;
    J[2] = in[2]*inp[0] + (in[0]-par[2])*inp[2];
}

void integrate(double *par, double *x0, double h, double tmax, double *out)
{
    double xp[3], fs[4][3], v[3];
    double h2, h6;
    int i, samples;

    // 4th order Runge-Kuta
    // t0 = 0
    xp[0] = x0[0];
    xp[1] = x0[1];
    xp[2] = x0[2];
    // printf( "%lf %lf %lf\n", xp[0], xp[1], xp[2] );

    h2 = h / 2.0;
    h6 = h / 6.0;
    samples = (int)(tmax / h + 0.5);
    for (i = 0; i < samples; i++)
    {
        v[0] = xp[0];
        v[1] = xp[1];
        v[2] = xp[2];
        sistema(&fs[0][0], v, par);

        v[0] = xp[0] + h2 * fs[0][0];
        v[1] = xp[1] + h2 * fs[0][1];
        v[2] = xp[2] + h2 * fs[0][2];
        sistema(&fs[1][0], v, par);

        v[0] = xp[0] + h2 * fs[1][0];
        v[1] = xp[1] + h2 * fs[1][1];
        v[2] = xp[2] + h2 * fs[1][2];
        sistema(&fs[2][0], v, par);

        v[0] = xp[0] + h * fs[2][0];
        v[1] = xp[1] + h * fs[2][1];
        v[2] = xp[2] + h * fs[2][2];
        sistema(&fs[3][0], v, par);

        xp[0] += h6 * (fs[0][0] + 2.0 * (fs[1][0] + fs[2][0]) + fs[3][0]);
        xp[1] += h6 * (fs[0][1] + 2.0 * (fs[1][1] + fs[2][1]) + fs[3][1]);
        xp[2] += h6 * (fs[0][2] + 2.0 * (fs[1][2] + fs[2][2]) + fs[3][2]);

#ifdef GRAFICA
        printf("%lf %lf %lf\n", xp[0], xp[1], xp[2]);
#endif
    }
    out[0] = xp[0];
    out[1] = xp[1];
    out[2] = xp[2];
}

void fullIntegration(double *par, double *x0, double h, double tmax, double *out)
{
    double xp[9], fs[4][9];
    double h2, h6;
    int i, j, samples;
    double v[9];

    // Este es un integrador Runge-Kuta de 4o orden
    // t0 = 0
    for (i = 0; i < 9; i++)
        xp[i] = x0[i];

    samples = (int)(tmax / h + 0.5);
    h2 = h / 2.0;
    h6 = h / 6.0;
    for (i = 0; i < samples; i++)
    {
        // 1
        for (j = 0; j < 9; j++)
            v[j] = xp[j];
        sistema(&fs[0][0], v, par);
        jacobiana(&fs[0][3], v, &v[3], par);
        jacobiana(&fs[0][6], v, &v[6], par);

        // 2
        for (j = 0; j < 9; j++)
            v[j] = xp[j] + h2 * fs[0][j];
        sistema(&fs[1][0], v, par);
        jacobiana(&fs[1][3], v, &v[3], par);
        jacobiana(&fs[1][6], v, &v[6], par);

        // 3
        for (j = 0; j < 9; j++)
            v[j] = xp[j] + h2 * fs[1][j];
        sistema(&fs[2][0], v, par);
        jacobiana(&fs[2][3], v, &v[3], par);
        jacobiana(&fs[2][6], v, &v[6], par);

        // 4
        for (j = 0; j < 9; j++)
            v[j] = xp[j] + h * fs[2][j];
        sistema(&fs[3][0], v, par);
        jacobiana(&fs[3][3], v, &v[3], par);
        jacobiana(&fs[3][6], v, &v[6], par);

        for (j = 0; j < 9; j++)
            xp[j] += h6 * (fs[0][j] + 2.0 * (fs[1][j] + fs[2][j]) + fs[3][j]);
    }

    for (j = 0; j < 9; j++)
        out[j] = xp[j];
}

void ortogonalizar(double *z, double *hatz)
{
    int i;
    double norma, val1, val2;
    double normaz2;

    for (i = 0; i < 3; i++)
        hatz[i] = z[i];

    norma = z[3] * z[3] + z[4] * z[4] + z[5] * z[5];
    normaz2 = sqrt(norma);
    hatz[3] = z[3] / normaz2;
    hatz[4] = z[4] / normaz2;
    hatz[5] = z[5] / normaz2;

    val1 = z[3] * z[6] + z[4] * z[7] + z[5] * z[8];
    val1 /= norma;
    hatz[6] = z[6] - z[3] * val1;
    hatz[7] = z[7] - z[4] * val1;
    hatz[8] = z[8] - z[5] * val1;

    val1 = z[3] * z[9] + z[4] * z[10] + z[5] * z[11];
    val1 /= norma;

    val2 = hatz[6] * z[9] + hatz[7] * z[10] + hatz[8] * z[11];
    normaz2 = hatz[6] * hatz[6] + hatz[7] * hatz[7] + hatz[8] * hatz[8];
    val2 /= normaz2;

    hatz[9] = z[9] - z[3] * val1 - hatz[6] * val2;
    hatz[10] = z[10] - z[4] * val1 - hatz[7] * val2;
    hatz[11] = z[11] - z[5] * val1 - hatz[8] * val2;

    norma = sqrt(normaz2);
    hatz[6] /= norma;
    hatz[7] /= norma;
    hatz[8] /= norma;

    norma = sqrt(hatz[9] * hatz[9] + hatz[10] * hatz[10] + hatz[11] * hatz[11]);
    hatz[9] /= norma;
    hatz[10] /= norma;
    hatz[11] /= norma;
}

void do_work(double *xini, double *par, double h, double t0, int itera, double tp)
{
    // El sistema es de 3er orden, se necesitan dos vectores
    double x0[12] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double hatx0[12];
    double v[12];
    int i, j;
    // int itera;
    // double h = 0.02;
    // double par[3] = { 16.0, 45.92, 4.0};
    // Valores de los parámetros para el sistema de Lorenz
    // double par[3] = { 10.0, 28.0, 2.666666666666 };
    // double par[3] = { 0.001, 0.001, .001 };
    double norm;
    double el[3];
    double dle1[3];
    double dle2[3];
    double dle3[3];

    // integrate( par, xini, h, 20.0,  v );
    integrate(par, xini, h, t0, v);
    for (i = 0; i < 3; i++)
        x0[i] = v[i];

    // itera = 1000;
    el[0] = el[1] = el[2] = 0.0;
    for (i = 0; i < itera; i++)
    {

        // fullIntegration( par, x0, h, 0.1, v );
        fullIntegration(par, x0, h, tp, v);
        // printf("%lf %lf %lf\n", v[0], v[1], v[2] );

        for (j = 9; j < 12; j++)
            v[j] = x0[j];
        ortogonalizar(v, hatx0);
        for (j = 0; j < 12; j++)
            v[j] = hatx0[j];

        jacobiana(dle1, v, &v[3], par);

        norm = v[3] * v[3] + v[4] * v[4] + v[5] * v[5];
        el[0] += (dle1[0] * v[3] + dle1[1] * v[4] + dle1[2] * v[5]) / norm;

        jacobiana(dle2, v, &v[6], par);

        norm = v[6] * v[6] + v[7] * v[7] + v[8] * v[8];
        el[1] += (dle2[0] * v[6] + dle2[1] * v[7] + dle2[2] * v[8]) / norm;

        jacobiana(dle3, v, &v[9], par);

        norm = v[9] * v[9] + v[10] * v[10] + v[11] * v[11];
        el[2] += (dle3[0] * v[9] + dle3[1] * v[10] + dle3[2] * v[11]) / norm;

        for (j = 0; j < 12; j++)
            x0[j] = v[j];
    }

    // exponentes calculados
    for (i = 0; i < 3; i++)
        el[i] /= itera;

    printf("%lf %lf %lf\n", el[0], el[1], el[2]);
}

int main(int argc, char *argv[])
{
    double x0[3], sigma, rho, beta;
    double h, t0, tp;
    int iteraciones;

    if (argc != 11)
    {
        fprintf(stderr, "Uso: %s x0 y0 z0 sigma rho beta h t0 iteraciones tp\n", argv[0]);
        return 1;
    }

    x0[0] = strtod(argv[1], NULL);
    x0[1] = strtod(argv[2], NULL);
    x0[2] = strtod(argv[3], NULL);
    sigma = strtod(argv[4], NULL);
    rho = strtod(argv[5], NULL);
    beta = strtod(argv[6], NULL);
    h = strtod(argv[7], NULL);
    t0 = strtod(argv[8], NULL);
    iteraciones = strtol(argv[9], NULL, 10);
    tp = strtod(argv[10], NULL);

    double par[3] = {sigma, rho, beta};

    do_work(x0, par, h, t0, iteraciones, tp);

    return 0;
}
