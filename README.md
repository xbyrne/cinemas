# CINEMAS :clapper:

**Constraining INclinations of Exoplanets and their MAsses by Stability**

![CINEMAS](./absolute_cinemas.png)

CINEMAS is a Bayesian framework for constraining the inclinations, and hence the true masses, of exoplanets in compact multi-planet systems detected with the radial velocity (RV) method.

The true mass $M$ of an exoplanet is not measured with the RV method, only the minimum mass $M_{\rm min}=M\sin i$, where $i$ is the (generally unknown) inclination angle. However, if $i$ were too low, this would mean the true masses of the planets ($M_j=M_{{\rm min}, j} / \sin i$) would be so big that the system would be dynamically unstable.

Assuming isotropy, the prior probability distribution on $i$ is $\pi(i)=\sin i$. The probability that a compact system with a given inclination (and hence given masses) is dynamically stable can be calculated quickly using the [`spock`](https://github.com/dtamayo/spock/) package. CINEMAS uses MCMC to calculate posterior distributions for the inclination, and thus the true masses of these exoplanets in inclined multi-planet systems.