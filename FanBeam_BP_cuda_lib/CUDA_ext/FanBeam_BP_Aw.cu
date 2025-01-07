#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_grid.cu"
#include "helper_math.cu"

#define CUDART_INF_F __int_as_float(0x7f800000)

texture<float, cudaTextureType2D, cudaReadModeElementType> volume_as_texture;
texture<float, cudaTextureType2D, cudaReadModeElementType> sinogram_as_texture;

cudaArray * VArray_Forward;
cudaArray * VArray_Backward;

float * ray_vec =NULL;

void FBP_Allocate_cu(const float * ray_vectors, const int volume_size_x,  const int volume_size_y, const int detector_size, const int number_of_projections)
{
    cudaMalloc((void**)&ray_vec, number_of_projections*2*sizeof(float));
    cudaMemcpy(ray_vec, ray_vectors, number_of_projections*2*sizeof(float), cudaMemcpyHostToDevice);

    // 1
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

    // 2.1
    volume_as_texture.addressMode[0]  = cudaAddressModeBorder;
    volume_as_texture.addressMode[1]  = cudaAddressModeBorder;
    volume_as_texture.filterMode      = cudaFilterModeLinear;
    volume_as_texture.normalized      = false;

    // 2.2
    cudaMallocArray(&VArray_Forward, &channelDesc, volume_size_x, volume_size_y);

    // 3.1
    sinogram_as_texture.addressMode[0] = cudaAddressModeBorder;
    sinogram_as_texture.addressMode[1] = cudaAddressModeBorder;
    sinogram_as_texture.filterMode = cudaFilterModeLinear;
    sinogram_as_texture.normalized = false;

    // 3.2
    cudaMallocArray(&VArray_Backward, &channelDesc, detector_size, number_of_projections);
}

void FBP_Free_cu()
{
    cudaFree(ray_vec);
    cudaFreeArray(VArray_Forward);
    cudaFreeArray(VArray_Backward);
}

/**************************************************************************************************/
__device__ float kernel_project2D(const float2 source_point, const float2 ray_vector, const float step_size, const int2 volume_size,
                                  const float2 volume_origin, const float2 volume_spacing)
{
    float pixel = 0.0f;
    // Step 1: compute alpha value at entry and exit point of the volume
    float min_alpha, max_alpha;
    min_alpha = 0;
    max_alpha = CUDART_INF_F;

    if (0.0f != ray_vector.x)
    {
        float volume_min_edge_point = index_to_physical(0, volume_origin.x, volume_spacing.x) - 0.5f;
        float volume_max_edge_point = index_to_physical(volume_size.x, volume_origin.x, volume_spacing.x) - 0.5f;

        float reci = 1.0f / ray_vector.x;
        float alpha0 = (volume_min_edge_point - source_point.x) * reci;
        float alpha1 = (volume_max_edge_point - source_point.x) * reci;
        min_alpha = fmin(alpha0, alpha1);
        max_alpha = fmax(alpha0, alpha1);
    }

    if (0.0f != ray_vector.y)
    {
        float volume_min_edge_point = index_to_physical(0, volume_origin.y, volume_spacing.y) - 0.5f;
        float volume_max_edge_point = index_to_physical(volume_size.y, volume_origin.y, volume_spacing.y) - 0.5f;

        float reci = 1.0f / ray_vector.y;
        float alpha0 = (volume_min_edge_point - source_point.y) * reci;
        float alpha1 = (volume_max_edge_point - source_point.y) * reci;
        min_alpha = fmax(min_alpha, fmin(alpha0, alpha1));
        max_alpha = fmin(max_alpha, fmax(alpha0, alpha1));
    }

    float px, py;
    //pixel = source_point.x + min_alpha * ray_vector.x;
    // Entrance boundary
    // In CUDA, voxel centers are located at (xx.5, xx.5, xx.5),
    //  whereas, SwVolume has voxel centers at integers.
    // For the initial interpolated value, only a half stepsize is
    //  considered in the computation.
    if (min_alpha < max_alpha)
    {
        px = source_point.x + min_alpha * ray_vector.x;
        py = source_point.y + min_alpha * ray_vector.y;

        pixel += 0.5f * tex2D(volume_as_texture, physical_to_index(px, volume_origin.x, volume_spacing.x) + 0.5f, physical_to_index(py, volume_origin.y, volume_spacing.y) + 0.5f);
        min_alpha += step_size;
    }
    // Mid segments
    while (min_alpha < max_alpha)
    {
        px = source_point.x + min_alpha * ray_vector.x;
        py = source_point.y + min_alpha * ray_vector.y;
        float2 interp_point = physical_to_index(make_float2(px, py), volume_origin, volume_spacing);
        pixel += tex2D(volume_as_texture, physical_to_index(px, volume_origin.x, volume_spacing.x) + 0.5f, physical_to_index(py, volume_origin.y, volume_spacing.y) + 0.5f);
        min_alpha += step_size;
    }
    // Scaling by stepsize;
    pixel *= step_size;

    // Last segment of the line
    if (pixel > 0.0f)
    {
        pixel -= 0.5f * step_size * tex2D(volume_as_texture, physical_to_index(px, volume_origin.x, volume_spacing.x) + 0.5f, physical_to_index(py, volume_origin.y, volume_spacing.y) + 0.5f);
        min_alpha -= step_size;
        float last_step_size = max_alpha - min_alpha;
        pixel += 0.5f * last_step_size * tex2D(volume_as_texture, physical_to_index(px, volume_origin.x, volume_spacing.x) + 0.5f, physical_to_index(py, volume_origin.y, volume_spacing.y) + 0.5f);

        px = source_point.x + max_alpha * ray_vector.x;
        py = source_point.y + max_alpha * ray_vector.y;
        // The last segment of the line integral takes care of the
        // varying length.
        pixel += 0.5f * last_step_size * tex2D(volume_as_texture, physical_to_index(px, volume_origin.x, volume_spacing.x) + 0.5f, physical_to_index(py, volume_origin.y, volume_spacing.y) + 0.5f);
    }
    return pixel;
}

__global__ void project_2Dfan_beam_kernel(float *pSinogram, const float2 *d_rays, const int number_of_projections, const float sampling_step_size,
                                          const int2 volume_size, const float volume_spacing_x, const float volume_spacing_y, const float volume_origin_x, const float volume_origin_y,
                                          const int detector_size, const float detector_spacing, const float detector_origin,
                                          const float sid, const float sdd)
{
    unsigned int detector_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (detector_idx >= detector_size)
    {
        return;
    }
    //Preparations:
    //Wrap pointer to float2 for better readable code
    float2 volume_spacing = make_float2(volume_spacing_x, volume_spacing_y);
    float2 volume_origin = make_float2(volume_origin_x, volume_origin_y);
    //Assume a source isocenter distance to compute the start of the ray, although sid is not neseccary for a par beam geometry
    //TODO: use volume spacing to reduce ray length
    int projection_idx = blockIdx.y;
    float2 central_ray_vector = d_rays[projection_idx];

    //create detector coordinate system (u,v) w.r.t (with respect to ) the ray
    float2 u_vec = make_float2(-central_ray_vector.y, central_ray_vector.x);
    //calculate physical coordinate of detector pixel
    float u = index_to_physical(detector_idx, detector_origin, detector_spacing);
    //Calculate "source"-Point (start point for the parallel ray), so we can use the projection kernel
    //Assume a source isocenter distance to compute the start of the ray, although sid is not neseccary for a par beam geometry
    float2 source_point = central_ray_vector * (-sid);

    float2 detector_point_world = source_point + central_ray_vector * sdd + u_vec * u;
    float2 ray_vector = normalize(detector_point_world - source_point);

    float pixel = kernel_project2D(source_point,
                                   ray_vector,
                                   sampling_step_size * fmin(volume_spacing.x, volume_spacing.y),
                                   volume_size,
                                   volume_origin,
                                   volume_spacing);

    pixel *= sqrt((ray_vector.x * volume_spacing.x) * (ray_vector.x * volume_spacing.x) + (ray_vector.y * volume_spacing.y) * (ray_vector.y * volume_spacing.y));

    unsigned sinogram_idx = projection_idx * detector_size + detector_idx;
    pSinogram[sinogram_idx] = pixel;

    return;
}

void FBP_2D_Aw_cu(at::Tensor input, const float * volume_ptr, float * out, //const float * ray_vectors,
                  const int number_of_projections, const int volume_size_x, const int volume_size_y, const float volume_spacing_x, const float volume_spacing_y,
                  const float volume_origin_x, const float volume_origin_y, const int detector_size, const float detector_spacing, const float detector_origin,
                  const float sid, const float sdd){
    // 1.texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    //volume_as_texture.addressMode[0]  = cudaAddressModeBorder;
    //volume_as_texture.addressMode[1]  = cudaAddressModeBorder;
    //volume_as_texture.filterMode      = cudaFilterModeLinear;
    //volume_as_texture.normalized      = false;

    //allocate and copy input tensor to cudaArray to be able to use the texture interpolation
    //cudaArray *volume_array;
    //cudaMallocArray(&volume_array, &channelDesc, volume_size_x, volume_size_y);
    cudaMemcpyToArray(VArray_Forward, 0, 0, volume_ptr, volume_size_x * volume_size_y * sizeof(float), cudaMemcpyHostToDevice);
    cudaBindTextureToArray(volume_as_texture, VArray_Forward, channelDesc);


    // 2.
    float sampling_step_size = 0.2;
    int2 volume_size = make_int2(volume_size_x, volume_size_y);

    const unsigned blocksize = 256;
    const dim3 gridsize = dim3((detector_size / blocksize) + 1, number_of_projections);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "project_2Dfan_beam_kernel", ([&] {
        project_2Dfan_beam_kernel<<<gridsize, blocksize>>>(out, ((float2 *) ray_vec), number_of_projections, sampling_step_size,
                                                           volume_size, volume_spacing_x, volume_spacing_y, volume_origin_x, volume_origin_y,
                                                           detector_size, detector_spacing, detector_origin,
                                                           sid, sdd);
        }));

    cudaUnbindTexture(volume_as_texture);
    //cudaFreeArray(volume_array);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in test1: %s\n", cudaGetErrorString(err));
}

/**************************************************************************************************/
inline __device__ float2 intersectLines2D(float2 p1, float2 p2, float2 p3, float2 p4)
{
    float dNom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);

    if (dNom < 0.000001f && dNom > -0.000001f)
    {
        float2 retValue = {NAN, NAN};
        return retValue;
    }
    float x = (p1.x * p2.y - p1.y * p2.x) * (p3.x - p4.x) - (p1.x - p2.x) * (p3.x * p4.y - p3.y * p4.x);
    float y = (p1.x * p2.y - p1.y * p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x * p4.y - p3.y * p4.x);

    x /= dNom;
    y /= dNom;
    float2 isectPt = {x, y};
    return isectPt;
}

__global__ void backproject_2Dfan_beam_kernel(float *pVolume, const float2 *d_rays, const int number_of_projections, const float sampling_step_size,
                                              const int2 volume_size, const float volume_spacing_x, const float volume_spacing_y, const float volume_origin_x, const float volume_origin_y,
                                              const int detector_size, const float detector_spacing, const float detector_origin,
                                              const float sid, const float sdd)
{
    const float pi = 3.14159265359f;
    unsigned int volume_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int volume_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (volume_x >= volume_size.x || volume_y >= volume_size.y)
    {
        return;
    }
    //Preparations:
    float2 volume_spacing = make_float2(volume_spacing_x, volume_spacing_y);
    float2 volume_origin = make_float2(volume_origin_x, volume_origin_y);
    const float2 pixel_coordinate = index_to_physical(make_float2(volume_x, volume_y), volume_origin, volume_spacing);
    float pixel_value = 0.0f;

    for (int n = 0; n < number_of_projections; n++)
    {
        float2 central_ray = d_rays[n];
        float2 detector_vec = make_float2(-central_ray.y, central_ray.x);

        float2 source_position = central_ray * (-sid);
        float2 central_point = source_position + central_ray * sdd;

        float2 intersection = intersectLines2D(pixel_coordinate, source_position, central_point, central_point + detector_vec);
        float distance_weight = 1.0f / (float)length(pixel_coordinate - source_position);
        float s = dot(intersection, detector_vec);
        float s_idx = physical_to_index(s, detector_origin, detector_spacing);

        pixel_value += tex2D(sinogram_as_texture, s_idx + 0.5f, n + 0.5f) * distance_weight * distance_weight;
    }

    const unsigned volume_linearized_idx = volume_y * volume_size.x + volume_x;
    pVolume[volume_linearized_idx] = sid * sdd * pi * pixel_value / number_of_projections;

    return;
}

void FBP_2D_Atw_cu(at::Tensor input, const float * sinogram_ptr, float * out, //const float * ray_vectors,
                  const int number_of_projections, const int volume_size_x, const int volume_size_y, const float volume_spacing_x, const float volume_spacing_y,
                  const float volume_origin_x, const float volume_origin_y, const int detector_size, const float detector_spacing, const float detector_origin,
                  const float sid, const float sdd){
    // 1.texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    //sinogram_as_texture.addressMode[0] = cudaAddressModeBorder;
    //sinogram_as_texture.addressMode[1] = cudaAddressModeBorder;
    //sinogram_as_texture.filterMode = cudaFilterModeLinear;
    //sinogram_as_texture.normalized = false;

    //cudaArray *sinogram_array;
    //cudaMallocArray(&sinogram_array, &channelDesc, detector_size, number_of_projections);
    cudaMemcpyToArray(VArray_Backward, 0, 0, sinogram_ptr, detector_size * number_of_projections * sizeof(float), cudaMemcpyHostToDevice);
    cudaBindTextureToArray(sinogram_as_texture, VArray_Backward, channelDesc);

    // 2.
    float sampling_step_size = 1;
    int2 volume_size = make_int2(volume_size_x, volume_size_y);

    const unsigned block_size = 16;
    const dim3 threads_per_block = dim3(block_size, block_size);
    const dim3 num_blocks = dim3(volume_size_x / threads_per_block.x + 1, volume_size_y / threads_per_block.y + 1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "backproject_2Dfan_beam_kernel", ([&] {
        backproject_2Dfan_beam_kernel<<<num_blocks, threads_per_block>>>(out, ((float2 *) ray_vec), number_of_projections, sampling_step_size,
                                                                         volume_size, volume_spacing_x, volume_spacing_y, volume_origin_x, volume_origin_y,
                                                                         detector_size, detector_spacing, detector_origin,
                                                                         sid, sdd);
        }));

    cudaUnbindTexture(volume_as_texture);
    //cudaFreeArray(sinogram_array);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in test1: %s\n", cudaGetErrorString(err));
}