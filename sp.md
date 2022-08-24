Alg.4 presents the classical Dijkstra algorithm,
which has been adopted in many researches, especially networking.
Here the classical Dijkstra algorithm is used to obtain the routing metric $G$.
The input parameters, i.e., the edge weights in the graph $W[N][N]$,
has been provided by the 16th Line of Alg.3.

**Alg.4 Delivery Metric based on Dijkstra from i to d**
>**Require**: $i$, $d$, $\mathcal{N}$, $W[N][N]$, $a$
>**Ensure**: $dis[d]$
>1. init distance vector $\{dis[x]\}_{1 \le x \le N}$ as *INF*
>2. init pointing vector $\{prev[x]\}_{1 \le x \le N}$ as $-1$
>3. init label vector $\{vis[x]\}_{1 \le x \le N}$ as $0$
>4. /* init the Dijkstra algorithm */
>5. **for** each $x$, $x \in \mathcal{N}$ **do**
>6.    $\quad dis[x] \leftarrow W[i][x]$
>7.    $\quad prev[x] \leftarrow i$
>8. **end for**
>9. $vis[i] \leftarrow 1$
>10. $count \leftarrow 1$
>11. **while** $count \neq N$ **do**
>12. $\quad$choose the minimum $x_m$ and $dis(x_m)$ from $\{dis[x]\}$
>13. $\quad vis[x_{m}] \leftarrow 1$
>14. $\quad count \leftarrow count+1$
>15. $\quad$ **for** each $r$, $r \in \mathcal{N}$ **do**
>16. $\quad$ $\quad$ **if** $vis[r] \neq 1$ and $W[x_m][i] \neq \inf$  
>$\quad$ $\quad$  $\quad$ $\quad$ and $dis[x_m] + W[x_m][r] < dis[r]$} **then**
>17. $\quad$ $\quad$ $\quad$ $dis[r] \leftarrow dis[x_m] + W[x_m][r]$
>18. $\quad$ $\quad$ $\quad$ $pre[r] \leftarrow x_m$
>19. $\quad$ $\quad$ **end if**
>20. $\quad$ **end for**
>21. **end while**


   
