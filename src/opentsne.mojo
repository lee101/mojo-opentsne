"""Numerical kernels for t-SNE.

All storage is owned by the caller. The C ABI wrappers rebuild typed pointers
from integer addresses so the exports themselves are not parametric.
"""

from std.algorithm import parallelize
from std.math import cos, exp, floor, iota, log, pow, sin, sqrt
from std.sys import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime PI = 3.1415926535897932384626433832795


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def ip32(addr: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=addr)


def ip64(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def kernel_weight(dist2: Float64, dof: Float64) -> Float64:
    if dof == 1.0:
        return 1.0 / (1.0 + dist2)
    return pow(1.0 + dist2 / dof, -dof)


def force_weight(dist2: Float64, dof: Float64) -> Float64:
    if dof == 1.0:
        var inv = 1.0 / (1.0 + dist2)
        return inv * inv
    return pow(1.0 + dist2 / dof, -(dof + 1.0))


def zero_buffer(data: Ptr, size: Int):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= size:
        data.store(i, SIMD[DType.float64, W](0.0))
        i += W
    while i < size:
        data[i] = 0.0
        i += 1


def scale_buffer(data: Ptr, size: Int, scale: Float64):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= size:
        var values = data.load[width=W](i)
        data.store(i, values * scale)
        i += W
    while i < size:
        data[i] *= scale
        i += 1


def sum_buffer(data: Ptr, size: Int) -> Float64:
    comptime W = simd_width_of[DType.float64]()
    var vector_total = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= size:
        vector_total += data.load[width=W](i)
        i += W
    var total = vector_total.reduce_add()
    while i < size:
        total += data[i]
        i += 1
    return total


def positive_gradient[
    IndexType: Intable & TrivialRegisterPassable
](
    indptr: UnsafePointer[IndexType, AnyOrigin[mut=True]],
    indices: UnsafePointer[IndexType, AnyOrigin[mut=True]],
    values: Ptr,
    embedding: Ptr,
    reference: Ptr,
    gradient: Ptr,
    n: Int,
    n_ref: Int,
    dim: Int,
    dof: Float64,
) -> Float64:
    var divergence = 0.0
    for i in range(n):
        var start = Int(indptr[i])
        var stop = Int(indptr[i + 1])
        for edge in range(start, stop):
            var j = Int(indices[edge])
            if j < 0 or j >= n_ref:
                continue
            var dist2 = 0.0
            for k in range(dim):
                var delta = embedding[i * dim + k] - reference[j * dim + k]
                dist2 += delta * delta
            var inv = 1.0 / (1.0 + dist2 / dof)
            var prob = values[edge]
            var coeff = prob * inv
            for k in range(dim):
                gradient[i * dim + k] += coeff * (
                    embedding[i * dim + k] - reference[j * dim + k]
                )
            if prob > 0.0:
                divergence += prob * (
                    log(prob) - log(kernel_weight(dist2, dof))
                )
    return divergence


def negative_exact(
    embedding: Ptr,
    reference: Ptr,
    gradient: Ptr,
    n: Int,
    n_ref: Int,
    dim: Int,
    dof: Float64,
    same_embedding: Bool,
) -> Float64:
    zero_buffer(gradient, n * dim)
    var sum_q = 0.0
    for i in range(n):
        for j in range(n_ref):
            if same_embedding and i == j:
                continue
            var dist2 = 2.220446049250313e-16
            for k in range(dim):
                var delta = embedding[i * dim + k] - reference[j * dim + k]
                dist2 += delta * delta
            var q = kernel_weight(dist2, dof)
            var f = force_weight(dist2, dof)
            sum_q += q
            for k in range(dim):
                gradient[i * dim + k] += f * (
                    embedding[i * dim + k] - reference[j * dim + k]
                )
        if not same_embedding:
            var row_q = 0.0
            for j in range(n_ref):
                var dist2 = 2.220446049250313e-16
                for k in range(dim):
                    var delta = embedding[i * dim + k] - reference[j * dim + k]
                    dist2 += delta * delta
                row_q += kernel_weight(dist2, dof)
            var row_scale = -1.0 / row_q if row_q > 0.0 else 0.0
            for k in range(dim):
                gradient[i * dim + k] *= row_scale
    if same_embedding:
        var scale = -1.0 / sum_q if sum_q > 0.0 else 0.0
        scale_buffer(gradient, n * dim, scale)
    return sum_q


def child_index(x: Float64, y: Float64, cx: Float64, cy: Float64) -> Int:
    return (1 if x > cx else 0) + (2 if y > cy else 0)


def init_node(
    node: Int,
    cx: Float64,
    cy: Float64,
    half: Float64,
    center_x: Ptr,
    center_y: Ptr,
    half_width: Ptr,
    mass: Ptr,
    sum_x: Ptr,
    sum_y: Ptr,
    point: Ptr,
    children: Ptr,
):
    center_x[node] = cx
    center_y[node] = cy
    half_width[node] = half
    mass[node] = 0.0
    sum_x[node] = 0.0
    sum_y[node] = 0.0
    point[node] = -1.0
    for q in range(4):
        children[node * 4 + q] = -1.0


def build_tree(
    embedding: Ptr,
    n: Int,
    capacity: Int,
    center_x: Ptr,
    center_y: Ptr,
    half_width: Ptr,
    mass: Ptr,
    sum_x: Ptr,
    sum_y: Ptr,
    point: Ptr,
    children: Ptr,
) -> Int:
    var lo_x = embedding[0]
    var hi_x = embedding[0]
    var lo_y = embedding[1]
    var hi_y = embedding[1]
    for i in range(1, n):
        var x = embedding[2 * i]
        var y = embedding[2 * i + 1]
        if x < lo_x:
            lo_x = x
        if x > hi_x:
            hi_x = x
        if y < lo_y:
            lo_y = y
        if y > hi_y:
            hi_y = y
    var root_half = 0.5 * max(hi_x - lo_x, hi_y - lo_y)
    init_node(
        0,
        0.5 * (lo_x + hi_x),
        0.5 * (lo_y + hi_y),
        root_half,
        center_x,
        center_y,
        half_width,
        mass,
        sum_x,
        sum_y,
        point,
        children,
    )
    var node_count = 1
    for i in range(n):
        var x = embedding[2 * i]
        var y = embedding[2 * i + 1]
        var node = 0
        var depth = 0
        while True:
            mass[node] += 1.0
            sum_x[node] += x
            sum_y[node] += y
            var first_child = Int(children[node * 4])
            if first_child < 0 and Int(point[node]) == -1:
                point[node] = Float64(i)
                break
            if first_child < 0:
                if depth >= 52:
                    point[node] = -2.0
                    break
                var previous_mass = mass[node] - 1.0
                var previous_x = (sum_x[node] - x) / previous_mass
                var previous_y = (sum_y[node] - y) / previous_mass
                if abs(previous_x - x) < 1e-6 and abs(previous_y - y) < 1e-6:
                    point[node] = -2.0
                    break
                if node_count + 4 > capacity:
                    return -1
                var old = Int(point[node])
                point[node] = -2.0
                var child_half = half_width[node] * 0.5
                for q in range(4):
                    var qx = center_x[node] + (
                        -child_half if q % 2 == 0 else child_half
                    )
                    var qy = center_y[node] + (
                        -child_half if q < 2 else child_half
                    )
                    init_node(
                        node_count + q,
                        qx,
                        qy,
                        child_half,
                        center_x,
                        center_y,
                        half_width,
                        mass,
                        sum_x,
                        sum_y,
                        point,
                        children,
                    )
                    children[node * 4 + q] = Float64(node_count + q)
                var old_x = previous_x
                var old_y = previous_y
                var old_q = child_index(
                    old_x, old_y, center_x[node], center_y[node]
                )
                var old_node = node_count + old_q
                mass[old_node] = 1.0
                sum_x[old_node] = old_x
                sum_y[old_node] = old_y
                point[old_node] = Float64(old) if old >= 0 else -2.0
                node_count += 4
            var q = child_index(x, y, center_x[node], center_y[node])
            node = Int(children[node * 4 + q])
            depth += 1
    return node_count


def negative_bh(
    embedding: Ptr,
    reference: Ptr,
    gradient: Ptr,
    n: Int,
    n_ref: Int,
    dof: Float64,
    theta: Float64,
    same_embedding: Bool,
    capacity: Int,
    center_x: Ptr,
    center_y: Ptr,
    half_width: Ptr,
    mass: Ptr,
    sum_x: Ptr,
    sum_y: Ptr,
    point: Ptr,
    children: Ptr,
    stack: Ptr,
) -> Float64:
    var node_count = build_tree(
        reference,
        n_ref,
        capacity,
        center_x,
        center_y,
        half_width,
        mass,
        sum_x,
        sum_y,
        point,
        children,
    )
    if node_count < 0:
        return -1.0
    var sum_q = 0.0
    var theta2 = theta * theta
    for i in range(n):
        var x = embedding[2 * i]
        var y = embedding[2 * i + 1]
        var fx = 0.0
        var fy = 0.0
        var row_q = 0.0
        var top = 1
        stack[0] = 0.0
        while top > 0:
            top -= 1
            var node = Int(stack[top])
            var node_mass = mass[node]
            if node_mass <= 0.0:
                continue
            var leaf = Int(children[node * 4]) < 0
            var effective_mass = node_mass
            var cx = sum_x[node] / node_mass
            var cy = sum_y[node] / node_mass
            if leaf:
                if same_embedding and Int(point[node]) == i:
                    continue
                if (
                    abs(cx - x) < 2.220446049250313e-16
                    and abs(cy - y) < 2.220446049250313e-16
                ):
                    continue
            var dx = x - cx
            var dy = y - cy
            var dist2 = dx * dx + dy * dy + 2.220446049250313e-16
            var width = 2.0 * half_width[node]
            if leaf or width * width < theta2 * dist2:
                var q = kernel_weight(dist2, dof)
                var f = force_weight(dist2, dof)
                row_q += effective_mass * q
                fx += effective_mass * f * dx
                fy += effective_mass * f * dy
            else:
                for q in range(4):
                    var child = Int(children[node * 4 + q])
                    if child >= 0:
                        stack[top] = Float64(child)
                        top += 1
        gradient[2 * i] = fx
        gradient[2 * i + 1] = fy
        sum_q += row_q
        if not same_embedding:
            var row_scale = -1.0 / row_q if row_q > 0.0 else 0.0
            gradient[2 * i] *= row_scale
            gradient[2 * i + 1] *= row_scale
    if same_embedding:
        var scale = -1.0 / sum_q if sum_q > 0.0 else 0.0
        scale_buffer(gradient, 2 * n, scale)
    return sum_q


def fft_line(data: Ptr, start: Int, stride: Int, n: Int, inverse: Bool):
    var j = 0
    for i in range(1, n):
        var bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            var ai = 2 * (start + i * stride)
            var aj = 2 * (start + j * stride)
            var tr = data[ai]
            var ti = data[ai + 1]
            data[ai] = data[aj]
            data[ai + 1] = data[aj + 1]
            data[aj] = tr
            data[aj + 1] = ti
    var length = 2
    while length <= n:
        var angle = (2.0 if inverse else -2.0) * PI / Float64(length)
        var wlen_r = cos(angle)
        var wlen_i = sin(angle)
        var block = 0
        while block < n:
            var wr = 1.0
            var wi = 0.0
            for k in range(length >> 1):
                var ia = 2 * (start + (block + k) * stride)
                var ib = 2 * (start + (block + k + (length >> 1)) * stride)
                var br = data[ib] * wr - data[ib + 1] * wi
                var bi = data[ib] * wi + data[ib + 1] * wr
                var ar = data[ia]
                var a_im = data[ia + 1]
                data[ia] = ar + br
                data[ia + 1] = a_im + bi
                data[ib] = ar - br
                data[ib + 1] = a_im - bi
                var next_wr = wr * wlen_r - wi * wlen_i
                wi = wr * wlen_i + wi * wlen_r
                wr = next_wr
            block += length
        length <<= 1
    if inverse:
        var scale = 1.0 / Float64(n)
        for i in range(n):
            data[2 * (start + i * stride)] *= scale
            data[2 * (start + i * stride) + 1] *= scale


def fft2(data: Ptr, side: Int, inverse: Bool):
    if side < 128:
        for row in range(side):
            fft_line(data, row * side, 1, side, inverse)
        for col in range(side):
            fft_line(data, col, side, side, inverse)
        return

    @parameter
    def transform_row(row: Int):
        fft_line(data, row * side, 1, side, inverse)

    @parameter
    def transform_col(col: Int):
        fft_line(data, col, side, side, inverse)

    parallelize[transform_row](side, 8)
    parallelize[transform_col](side, 8)


def grid_coordinates(
    embedding: Ptr,
    i: Int,
    origin_x: Float64,
    origin_y: Float64,
    spacing: Float64,
) -> Tuple[Int, Int, Float64, Float64]:
    var gx = (embedding[2 * i] - origin_x) / spacing
    var gy = (embedding[2 * i + 1] - origin_y) / spacing
    var ix = Int(floor(gx))
    var iy = Int(floor(gy))
    return ix, iy, gx - Float64(ix), gy - Float64(iy)


def deposit_grid(
    embedding: Ptr,
    n: Int,
    grid: Ptr,
    side: Int,
    origin_x: Float64,
    origin_y: Float64,
    spacing: Float64,
):
    zero_buffer(grid, 2 * side * side)
    for i in range(n):
        var ix, iy, tx, ty = grid_coordinates(
            embedding, i, origin_x, origin_y, spacing
        )
        var w00 = (1.0 - tx) * (1.0 - ty)
        var w10 = tx * (1.0 - ty)
        var w01 = (1.0 - tx) * ty
        var w11 = tx * ty
        grid[2 * (iy * side + ix)] += w00
        grid[2 * (iy * side + ix + 1)] += w10
        grid[2 * ((iy + 1) * side + ix)] += w01
        grid[2 * ((iy + 1) * side + ix + 1)] += w11


def make_kernel_row(
    grid: Ptr,
    side: Int,
    spacing: Float64,
    dof: Float64,
    component: Int,
    iy: Int,
):
    var sy = iy if iy <= side // 2 else iy - side
    var dy = Float64(sy) * spacing
    var row_offset = 2 * iy * side
    if dof != 1.0:
        for ix in range(side):
            var sx = ix if ix <= side // 2 else ix - side
            var dx = Float64(sx) * spacing
            var dist2 = dx * dx + dy * dy
            var value = kernel_weight(dist2, dof)
            if component == 1:
                value = dx * force_weight(dist2, dof)
            elif component == 2:
                value = dy * force_weight(dist2, dof)
            var offset = row_offset + 2 * ix
            grid[offset] = value
            grid[offset + 1] = 0.0
        return

    comptime W = simd_width_of[DType.float64]()
    var ix = 0
    var positive_stop = side // 2 + 1
    while ix + W <= positive_stop:
        var dx_positive = (iota[DType.float64, W](Float64(ix))) * spacing
        var inv_positive = 1.0 / (1.0 + dx_positive * dx_positive + dy * dy)
        var value_positive = inv_positive
        if component == 1:
            value_positive = dx_positive * inv_positive * inv_positive
        elif component == 2:
            value_positive = dy * inv_positive * inv_positive
        var complex_positive = value_positive.interleave(
            SIMD[DType.float64, W](0.0)
        )
        grid.store(row_offset + 2 * ix, complex_positive)
        ix += W
    while ix < positive_stop:
        var dx_positive_tail = Float64(ix) * spacing
        var inv_positive_tail = 1.0 / (
            1.0 + dx_positive_tail * dx_positive_tail + dy * dy
        )
        var value_positive_tail = inv_positive_tail
        if component == 1:
            value_positive_tail = (
                dx_positive_tail * inv_positive_tail * inv_positive_tail
            )
        elif component == 2:
            value_positive_tail = dy * inv_positive_tail * inv_positive_tail
        grid[row_offset + 2 * ix] = value_positive_tail
        grid[row_offset + 2 * ix + 1] = 0.0
        ix += 1
    while ix + W <= side:
        var dx_negative = (iota[DType.float64, W](Float64(ix - side))) * spacing
        var inv_negative = 1.0 / (1.0 + dx_negative * dx_negative + dy * dy)
        var value_negative = inv_negative
        if component == 1:
            value_negative = dx_negative * inv_negative * inv_negative
        elif component == 2:
            value_negative = dy * inv_negative * inv_negative
        var complex_negative = value_negative.interleave(
            SIMD[DType.float64, W](0.0)
        )
        grid.store(row_offset + 2 * ix, complex_negative)
        ix += W
    while ix < side:
        var dx_negative_tail = Float64(ix - side) * spacing
        var inv_negative_tail = 1.0 / (
            1.0 + dx_negative_tail * dx_negative_tail + dy * dy
        )
        var value_negative_tail = inv_negative_tail
        if component == 1:
            value_negative_tail = (
                dx_negative_tail * inv_negative_tail * inv_negative_tail
            )
        elif component == 2:
            value_negative_tail = dy * inv_negative_tail * inv_negative_tail
        grid[row_offset + 2 * ix] = value_negative_tail
        grid[row_offset + 2 * ix + 1] = 0.0
        ix += 1


def make_kernel(
    grid: Ptr, side: Int, spacing: Float64, dof: Float64, component: Int
):
    for iy in range(side):
        make_kernel_row(grid, side, spacing, dof, component, iy)


def multiply_spectrum(density: Ptr, field: Ptr, cells: Int):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= cells:
        var density_values = density.load[width=2 * W](2 * i)
        var field_values = field.load[width=2 * W](2 * i)
        var ar, ai = density_values.deinterleave()
        var br, bi = field_values.deinterleave()
        var real = ar * br - ai * bi
        var imag = ar * bi + ai * br
        field.store(2 * i, real.interleave(imag))
        i += W
    while i < cells:
        var ar = density[2 * i]
        var ai = density[2 * i + 1]
        var br = field[2 * i]
        var bi = field[2 * i + 1]
        field[2 * i] = ar * br - ai * bi
        field[2 * i + 1] = ar * bi + ai * br
        i += 1


def interpolate_field(
    embedding: Ptr,
    n: Int,
    field: Ptr,
    gradient: Ptr,
    side: Int,
    origin_x: Float64,
    origin_y: Float64,
    spacing: Float64,
    component: Int,
    dof: Float64,
) -> Float64:
    var total = 0.0
    for i in range(n):
        var ix, iy, tx, ty = grid_coordinates(
            embedding, i, origin_x, origin_y, spacing
        )
        var w00 = (1.0 - tx) * (1.0 - ty)
        var w10 = tx * (1.0 - ty)
        var w01 = (1.0 - tx) * ty
        var w11 = tx * ty
        var value = (
            w00 * field[2 * (iy * side + ix)]
            + w10 * field[2 * (iy * side + ix + 1)]
            + w01 * field[2 * ((iy + 1) * side + ix)]
            + w11 * field[2 * ((iy + 1) * side + ix + 1)]
        )
        if component == 0:
            var wx0 = 1.0 - tx
            var wx1 = tx
            var wy0 = 1.0 - ty
            var wy1 = ty
            var self_value = 0.0
            for ay in range(2):
                var way = wy0 if ay == 0 else wy1
                for ax in range(2):
                    var wax = wx0 if ax == 0 else wx1
                    for by in range(2):
                        var wby = wy0 if by == 0 else wy1
                        for bx in range(2):
                            var wbx = wx0 if bx == 0 else wx1
                            var dx = Float64(ax - bx) * spacing
                            var dy = Float64(ay - by) * spacing
                            self_value += (
                                wax
                                * way
                                * wbx
                                * wby
                                * kernel_weight(dx * dx + dy * dy, dof)
                            )
            total += value - self_value
        else:
            gradient[2 * i + component - 1] = value
    return total


def negative_fft(
    embedding: Ptr,
    gradient: Ptr,
    density: Ptr,
    field: Ptr,
    n: Int,
    side: Int,
    origin_x: Float64,
    origin_y: Float64,
    spacing: Float64,
    dof: Float64,
) -> Float64:
    deposit_grid(embedding, n, density, side, origin_x, origin_y, spacing)
    fft2(density, side, False)
    for component in range(1, 3):
        make_kernel(field, side, spacing, dof, component)
        fft2(field, side, False)
        multiply_spectrum(density, field, side * side)
        fft2(field, side, True)
        _ = interpolate_field(
            embedding,
            n,
            field,
            gradient,
            side,
            origin_x,
            origin_y,
            spacing,
            component,
            dof,
        )
    make_kernel(field, side, spacing, dof, 0)
    fft2(field, side, False)
    multiply_spectrum(density, field, side * side)
    fft2(field, side, True)
    var sum_q = interpolate_field(
        embedding,
        n,
        field,
        gradient,
        side,
        origin_x,
        origin_y,
        spacing,
        0,
        dof,
    )
    var scale = -1.0 / sum_q if sum_q > 0.0 else 0.0
    scale_buffer(gradient, 2 * n, scale)
    return sum_q


def perplexity_probabilities(
    distances2: Ptr, probabilities: Ptr, n: Int, k: Int, perplexity: Float64
):
    var target = log(perplexity)
    for i in range(n):
        var beta = 1.0
        var beta_min = 0.0
        var beta_max = 1.7976931348623157e308
        var sum_p: Float64
        for iteration in range(100):
            sum_p = 0.0
            var weighted = 0.0
            for j in range(k):
                var value = exp(-beta * distances2[i * k + j])
                probabilities[i * k + j] = value
                sum_p += value
                weighted += distances2[i * k + j] * value
            var entropy = log(sum_p) + beta * weighted / sum_p
            var delta = entropy - target
            if abs(delta) < 1e-8:
                break
            if delta > 0.0:
                beta_min = beta
                beta = (
                    beta * 2.0 if beta_max
                    == 1.7976931348623157e308 else 0.5 * (beta + beta_max)
                )
            else:
                beta_max = beta
                beta = 0.5 * beta if beta_min == 0.0 else 0.5 * (
                    beta + beta_min
                )
        sum_p = 0.0
        for j in range(k):
            sum_p += probabilities[i * k + j]
        for j in range(k):
            probabilities[i * k + j] /= sum_p


def multiscale_probabilities(
    distances2: Ptr,
    probabilities: Ptr,
    perplexities: Ptr,
    n: Int,
    k: Int,
    n_scales: Int,
):
    for i in range(n):
        for j in range(k):
            probabilities[i * k + j] = 0.0
        var beta_min = 0.0
        var beta_max = 1.7976931348623157e308
        for scale in range(n_scales):
            var beta = 0.5
            var target = log(perplexities[scale])
            for iteration in range(200):
                var sum_p = 0.0
                var weighted = 0.0
                for j in range(k):
                    var value = sqrt(2.0 * beta) * exp(
                        -beta * distances2[i * k + j]
                    )
                    sum_p += value
                    weighted += distances2[i * k + j] * value
                var entropy = (
                    beta * weighted / sum_p
                    + log(sum_p)
                    - 0.5 * log(2.0 * beta)
                )
                var delta = entropy - target
                if abs(delta) <= 1e-8:
                    break
                if delta > 0.0:
                    beta_min = beta
                    beta = (
                        beta * 2.0 if beta_max
                        == 1.7976931348623157e308 else 0.5 * (beta + beta_max)
                    )
                else:
                    beta_max = beta
                    beta = 0.5 * beta if beta_min == 0.0 else 0.5 * (
                        beta + beta_min
                    )
            for j in range(k):
                probabilities[i * k + j] += sqrt(2.0 * beta) * exp(
                    -beta * distances2[i * k + j]
                )
        var total = 0.0
        for j in range(k):
            total += probabilities[i * k + j]
        for j in range(k):
            probabilities[i * k + j] /= total


@export("mot_positive_gradient")
def mot_positive_gradient(
    indptr_addr: Int,
    indices_addr: Int,
    values_addr: Int,
    embedding_addr: Int,
    reference_addr: Int,
    gradient_addr: Int,
    n: Int,
    n_ref: Int,
    dim: Int,
    dof: Float64,
) abi("C") -> Float64:
    return positive_gradient(
        ip64(indptr_addr),
        ip64(indices_addr),
        p(values_addr),
        p(embedding_addr),
        p(reference_addr),
        p(gradient_addr),
        n,
        n_ref,
        dim,
        dof,
    )


@export("mot_positive_gradient_i32")
def mot_positive_gradient_i32(
    indptr_addr: Int,
    indices_addr: Int,
    values_addr: Int,
    embedding_addr: Int,
    reference_addr: Int,
    gradient_addr: Int,
    n: Int,
    n_ref: Int,
    dim: Int,
    dof: Float64,
) abi("C") -> Float64:
    return positive_gradient(
        ip32(indptr_addr),
        ip32(indices_addr),
        p(values_addr),
        p(embedding_addr),
        p(reference_addr),
        p(gradient_addr),
        n,
        n_ref,
        dim,
        dof,
    )


@export("mot_negative_exact")
def mot_negative_exact(
    embedding_addr: Int,
    reference_addr: Int,
    gradient_addr: Int,
    n: Int,
    n_ref: Int,
    dim: Int,
    dof: Float64,
    same_embedding: Int,
) abi("C") -> Float64:
    return negative_exact(
        p(embedding_addr),
        p(reference_addr),
        p(gradient_addr),
        n,
        n_ref,
        dim,
        dof,
        same_embedding != 0,
    )


@export("mot_negative_bh")
def mot_negative_bh(
    embedding_addr: Int,
    reference_addr: Int,
    gradient_addr: Int,
    n: Int,
    n_ref: Int,
    dof: Float64,
    theta: Float64,
    same_embedding: Int,
    capacity: Int,
    center_x_addr: Int,
    center_y_addr: Int,
    half_width_addr: Int,
    mass_addr: Int,
    sum_x_addr: Int,
    sum_y_addr: Int,
    point_addr: Int,
    children_addr: Int,
    stack_addr: Int,
) abi("C") -> Float64:
    return negative_bh(
        p(embedding_addr),
        p(reference_addr),
        p(gradient_addr),
        n,
        n_ref,
        dof,
        theta,
        same_embedding != 0,
        capacity,
        p(center_x_addr),
        p(center_y_addr),
        p(half_width_addr),
        p(mass_addr),
        p(sum_x_addr),
        p(sum_y_addr),
        p(point_addr),
        p(children_addr),
        p(stack_addr),
    )


@export("mot_negative_fft")
def mot_negative_fft(
    embedding_addr: Int,
    gradient_addr: Int,
    density_addr: Int,
    field_addr: Int,
    n: Int,
    side: Int,
    origin_x: Float64,
    origin_y: Float64,
    spacing: Float64,
    dof: Float64,
) abi("C") -> Float64:
    return negative_fft(
        p(embedding_addr),
        p(gradient_addr),
        p(density_addr),
        p(field_addr),
        n,
        side,
        origin_x,
        origin_y,
        spacing,
        dof,
    )


@export("mot_deposit_grid")
def mot_deposit_grid(
    embedding_addr: Int,
    grid_addr: Int,
    n: Int,
    side: Int,
    origin_x: Float64,
    origin_y: Float64,
    spacing: Float64,
) abi("C"):
    deposit_grid(
        p(embedding_addr),
        n,
        p(grid_addr),
        side,
        origin_x,
        origin_y,
        spacing,
    )


@export("mot_make_kernel")
def mot_make_kernel(
    grid_addr: Int,
    side: Int,
    spacing: Float64,
    dof: Float64,
    component: Int,
) abi("C"):
    make_kernel(p(grid_addr), side, spacing, dof, component)


@export("mot_multiply_spectrum")
def mot_multiply_spectrum(
    density_addr: Int,
    field_addr: Int,
    cells: Int,
) abi("C"):
    multiply_spectrum(p(density_addr), p(field_addr), cells)


@export("mot_interpolate_field")
def mot_interpolate_field(
    embedding_addr: Int,
    field_addr: Int,
    gradient_addr: Int,
    n: Int,
    side: Int,
    origin_x: Float64,
    origin_y: Float64,
    spacing: Float64,
    component: Int,
    dof: Float64,
) abi("C") -> Float64:
    return interpolate_field(
        p(embedding_addr),
        n,
        p(field_addr),
        p(gradient_addr),
        side,
        origin_x,
        origin_y,
        spacing,
        component,
        dof,
    )


@export("mot_scale_buffer")
def mot_scale_buffer(
    data_addr: Int,
    size: Int,
    scale: Float64,
) abi("C"):
    scale_buffer(p(data_addr), size, scale)


@export("mot_sum_buffer")
def mot_sum_buffer(data_addr: Int, size: Int) abi("C") -> Float64:
    return sum_buffer(p(data_addr), size)


@export("mot_perplexity_probabilities")
def mot_perplexity_probabilities(
    distances_addr: Int,
    probabilities_addr: Int,
    n: Int,
    k: Int,
    perplexity: Float64,
) abi("C"):
    perplexity_probabilities(
        p(distances_addr), p(probabilities_addr), n, k, perplexity
    )


@export("mot_multiscale_probabilities")
def mot_multiscale_probabilities(
    distances_addr: Int,
    probabilities_addr: Int,
    perplexities_addr: Int,
    n: Int,
    k: Int,
    n_scales: Int,
) abi("C"):
    multiscale_probabilities(
        p(distances_addr),
        p(probabilities_addr),
        p(perplexities_addr),
        n,
        k,
        n_scales,
    )
