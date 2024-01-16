import numpy as np

def concentric_mapping_square_to_hemisphere(n_samples_per_axis: int):

    N = n_samples_per_axis
    R1 = np.linspace(0.01, 0.99, N)
    R2 = np.linspace(0.01, 0.99, N)

    # % Initial mapping
    phi = 0.
    radius = 0.
    a = (2 * R1) - 1
    b = (2 * R2) - 1
    a_mesh, b_mesh = np.meshgrid(a, b, indexing='ij')

    u_mesh = np.zeros_like(a_mesh)
    v_mesh = np.zeros_like(a_mesh)
    w_mesh = np.zeros_like(a_mesh)

    for i in range(N):
        for j in range(N):
            a = a_mesh[i, j]
            b = b_mesh[i, j]
            if a > -b:
                if a > b:
                    radius = a
                    phi = (np.pi/4) * (b/a)        
                else:
                    radius = b
                    phi = (np.pi/4) * (2. - (a/b))
            else:
                if a < b:
                    radius = -a
                    phi = (np.pi/4) * (4. + (b/a))
                else:
                    radius = -b
                    if b != 0.:
                        phi = (np.pi/4) * (6. - (a/b))
                    else:
                        phi = 0.
            u = np.cos(phi) * radius
            v = np.sin(phi) * radius

            x = u * np.sqrt(2 - radius**2)
            y = v * np.sqrt(2 - radius**2)
            z = 1 - (radius**2)
            u_mesh[i, j] = x
            v_mesh[i, j] = y
            w_mesh[i, j] = z

    points = np.stack((u_mesh, v_mesh, w_mesh), axis=2)
    
    # points: [x, y, 3]
    return points
