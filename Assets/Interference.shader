Shader "Unlit/Interference"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
		iChannel0 ("Texture", Cube) = "white" {}
		iChannel1("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

			//#define REFLECTANCE_ONLY

			// performance and raymarching options
			#define INTERSECTION_PRECISION 0.01  // raymarcher intersection precision
			#define ITERATIONS 20				 // max number of iterations
			#define AA_SAMPLES 1				 // anti aliasing samples
			#define BOUND 6.0					 // cube bounds check
			#define DIST_SCALE 0.9   			 // scaling factor for raymarching position update

			// optical properties
			#define DISPERSION 0.05				 // dispersion amount
			#define IOR 0.9     				 // base IOR value specified as a ratio
			#define THICKNESS_SCALE 32.0		 // film thickness scaling factor
			#define THICKNESS_CUBEMAP_SCALE 0.1  // film thickness cubemap scaling factor
			#define REFLECTANCE_SCALE 3.0        // reflectance scaling factor
			#define REFLECTANCE_GAMMA_SCALE 2.0  // reflectance gamma scaling factor
			#define FRESNEL_RATIO 0.7			 // fresnel weight for reflectance
			#define SIGMOID_CONTRAST 4.0       //8.0  // contrast enhancement

			#define TWO_PI 6.28318530718
			#define WAVELENGTHS 6				 // number of wavelengths, not a free parameter

			samplerCUBE iChannel0;
			sampler2D iChannel1;

			// iq's cubemap function
			float3 fancyCube(sampler2D sam, in float3 d, in float s, in float b)
			{
				float4 locationX = ((0.5 + s * d.y) / d.x, (0.5 + s * d.z) / d.x, 0, b);
				float4 colX = tex2Dlod(sam, locationX);
				float3 colx = colX.xyz;
				float4 locationY = (0.5 + s * d.z / d.y, 0.5 + s * d.x / d.y, 0, b);
				float3 coly = tex2Dlod(sam, locationY).xyz;
				float4 locationZ = (0.5 + s * d.x / d.z, 0.5 + s * d.y / d.z, 0, b);
				float3 colz = tex2Dlod(sam, locationZ).xyz;

				float3 n = d * d;

				return (colx*n.x + coly * n.y + colz * n.z) / (n.x + n.y + n.z);
			}

			// iq's 3D noise function
			float hash(float n) {
				return frac(sin(n)*43758.5453);
			}

			float noise(in float3 x) {
				float3 p = floor(x);
				float3 f = frac(x);

				f = f * f*(3.0 - 2.0*f);
				float n = p.x + p.y*57.0 + 113.0*p.z;
				return lerp(lerp(lerp(hash(n + 0.0), hash(n + 1.0), f.x),
					lerp(hash(n + 57.0), hash(n + 58.0), f.x), f.y),
					lerp(lerp(hash(n + 113.0), hash(n + 114.0), f.x),
						lerp(hash(n + 170.0), hash(n + 171.0), f.x), f.y), f.z);
			}

			float3 noise3(float3 x) {
				return float3(noise(x + float3(123.456, .567, .37)),
					noise(x + float3(.11, 47.43, 19.17)),
					noise(x));
			}

			// a sphere with a little bit of warp
			float sdf(float3 p) {
				float3 n = float3(sin(_Time.y * 0.5), sin(_Time.y * 0.3), cos(_Time.y * 0.2));
				float3 q = 0.1 * (noise3(p + n) - 0.5);

				return length(q + p) - 3.5;
			}

			float3 fresnel(float3 rd, float3 norm, float3 n2) {
				float3 r0 = pow((1.0 - n2) / (1.0 + n2), float3(2, 2, 2));
				return r0 + (1. - r0)*pow(clamp(1. + dot(rd, norm), 0.0, 1.0), 5.);
			}

			float3 calcNormal(in float3 pos) {
				static const float eps = INTERSECTION_PRECISION;

				static const float3 v1 = float3(1.0, -1.0, -1.0);
				static const float3 v2 = float3(-1.0, -1.0, 1.0);
				static const float3 v3 = float3(-1.0, 1.0, -1.0);
				static const float3 v4 = float3(1.0, 1.0, 1.0);

				return normalize(v1*sdf(pos + v1 * eps) +
					v2 * sdf(pos + v2 * eps) +
					v3 * sdf(pos + v3 * eps) +
					v4 * sdf(pos + v4 * eps));
			}

			#define GAMMA_CURVE 50.0
			#define GAMMA_SCALE 4.5
			float3 filmic_gamma(float3 x) {
				return log(GAMMA_CURVE * x + 1.0) / GAMMA_SCALE;
			}

			float3 filmic_gamma_inverse(float3 y) {
				return (1.0 / GAMMA_CURVE) * (exp(GAMMA_SCALE * y) - 1.0);
			}

			// sample weights for the cubemap given a wavelength i
			// room for improvement in this function
			#define GREEN_WEIGHT 2.8
			float3 texCubeSampleWeights(float i) {
				float3 w = float3((1.0 - i) * (1.0 - i), GREEN_WEIGHT * i * (1.0 - i), i * i);
				return w / dot(w, float3(1.0, 1.0, 1.0));
			}

			float3 sampleCubeMap(float3 i, float3 rd) {
				//float4 location = (rd, 0);
				//float4 colI = texCUBElod(iChannel0, location);
				//float3 col = texCUBE(iChannel0, rd).xyz;
				//float3 col = texCUBElod(iChannel0, location).xyz;
				
				return texCUBE(iChannel0, rd).xyz;
				//return col;
				/*
				return float3(
					dot(texCubeSampleWeights(i.x), col),
					dot(texCubeSampleWeights(i.y), col),
					dot(texCubeSampleWeights(i.z), col)
				);
				*/	
			}
			
			float3 sampleCubeMap(float3 i, float3 rd0, float3 rd1, float3 rd2) {
				/*
				float4 location0 = (rd0 * float3(1.0, -1.0, 1.0), 0.0);
				float4 location1 = (rd1 * float3(1.0, -1.0, 1.0), 0.0);
				float4 location2 = (rd2 * float3(1.0, -1.0, 1.0), 0.0);
				float3 col0 = texCUBElod(iChannel0, location0).xyz;
				float3 col1 = texCUBElod(iChannel0, location1).xyz;
				float3 col2 = texCUBElod(iChannel0, location2).xyz;
				*/
				float3 col0 = texCUBE(iChannel0, rd0).xyz;
				float3 col1 = texCUBE(iChannel0, rd1).xyz;
				float3 col2 = texCUBE(iChannel0, rd2).xyz;
				
				return float3(
					dot(texCubeSampleWeights(i.x), col0),
					dot(texCubeSampleWeights(i.y), col1),
					dot(texCubeSampleWeights(i.z), col2)
				);
			}
			

			float3 sampleWeights(float i) {
				return float3((1.0 - i) * (1.0 - i), GREEN_WEIGHT * i * (1.0 - i), i * i);
			}

			float3 resample(float3 wl0, float3 wl1, float3 i0, float3 i1) {
				float3 w0 = sampleWeights(wl0.x);
				float3 w1 = sampleWeights(wl0.y);
				float3 w2 = sampleWeights(wl0.z);
				float3 w3 = sampleWeights(wl1.x);
				float3 w4 = sampleWeights(wl1.y);
				float3 w5 = sampleWeights(wl1.z);

				return i0.x * w0 + i0.y * w1 + i0.z * w2
					+ i1.x * w3 + i1.y * w4 + i1.z * w5;
			}

			// downsample to RGB
			float3 resampleColor(float3 rds[WAVELENGTHS], float3 refl0, float3 refl1, float3 wl0, float3 wl1) {

				#ifdef REFLECTANCE_ONLY
				float3 intensity0 = refl0;
				float3 intensity1 = refl1;
				#else
				float3 cube0 = sampleCubeMap(wl0, rds[0], rds[1], rds[2]);
				float3 cube1 = sampleCubeMap(wl1, rds[3], rds[4], rds[5]);

				float3 intensity0 = filmic_gamma_inverse(cube0) + refl0;
				float3 intensity1 = filmic_gamma_inverse(cube1) + refl1;
				#endif
				float3 col = resample(wl0, wl1, intensity0, intensity1);

				return 1.4 * filmic_gamma(col / float(WAVELENGTHS));
			}

			float3 resampleColorSimple(float3 rd, float3 wl0, float3 wl1) {
				float3 cube0 = sampleCubeMap(wl0, rd);
				float3 cube1 = sampleCubeMap(wl1, rd);

				float3 intensity0 = filmic_gamma_inverse(cube0);
				float3 intensity1 = filmic_gamma_inverse(cube1);
				float3 col = resample(wl0, wl1, intensity0, intensity1);
				return cube0;
				return 1.4 * filmic_gamma(col / float(WAVELENGTHS));
			}

			// compute the wavelength/IOR curve values.
			float3 iorCurve(float3 x) {
				return x;
			}

			float3 attenuation(float filmThickness, float3 wavelengths, float3 normal, float3 rd) {
				return 0.5 + 0.5 * cos(((THICKNESS_SCALE * filmThickness) / (wavelengths + 1.0)) * dot(normal, rd));
			}

			float3 contrast(float3 x) {
				return 1.0 / (1.0 + exp(-SIGMOID_CONTRAST * (x - 0.5)));
			}

			void doCamera(out float3 camPos, out float3 camTar, in float time, in float4 m) {
				camTar = float3(0.0, 0.0, 0.0);
				if (max(m.z, m.w) <= 0.0) {
					float an = 1.5 + sin(time * 0.05) * 4.0;
					camPos = float3(6.5*sin(an), -2.0*cos(an), 6.5*cos(an));
				}
				else {
					float an = 10.0 * m.x - 5.0;
					camPos = float3(6.5*sin(an), 10.0 * m.y - 5.0, 6.5*cos(an));
				}
			}

			float3x3 calcLookAtMatrix(in float3 ro, in float3 ta, in float roll)
			{
				float3 ww = normalize(ta - ro);
				float3 uu = normalize(cross(ww, float3(sin(roll), cos(roll), 0.0)));
				float3 vv = normalize(cross(uu, ww));
				return float3x3(uu, vv, ww);
			}

            fixed4 frag (v2f i) : SV_Target
            {
				//vec2 p = (-iResolution.xy + 2.0*fragCoord.xy) / iResolution.y;
				//vec4 m = vec4(iMouse.xy / iResolution.xy, iMouse.zw);
				float2 p = (-_ScreenParams.xy + (2.0* _ScreenParams.xy*i.uv)) / _ScreenParams.y;
				//float2 p = i.uv;
				float4 m = float4(0.0, 0.0, 0.0, 0.0);

				// camera movement
				float3 ro, ta;
				doCamera(ro, ta, _Time.y, m);
				float3x3 camMat = calcLookAtMatrix(ro, ta, 0.0);

				float dh = (0.666 / _ScreenParams.y);
				static const float rads = TWO_PI / float(AA_SAMPLES);

				float3 col = float3(0.0, 0.0, 0.0);

				static const float3 wavelengths0 = float3(1.0, 0.8, 0.6);
				static const float3 wavelengths1 = float3(0.4, 0.2, 0.0);
				float3 iors0 = IOR + iorCurve(wavelengths0) * DISPERSION;
				float3 iors1 = IOR + iorCurve(wavelengths1) * DISPERSION;

				float3 rds[WAVELENGTHS];

				for (int samp = 0; samp < AA_SAMPLES; samp++) {
					float2 dxy = dh * float2(cos(float(samp) * rads), sin(float(samp) * rads));
					//float3 rd = normalize(camMat * float3(p.xy + dxy, 1.5)); // 1.5 is the lens length
					float3 rd = normalize(mul(float3(p.xy + dxy, 1.1), camMat));
					float3 pos = ro;
					bool hit = false;
					for (int j = 0; j < ITERATIONS; j++) {
						float t = DIST_SCALE * sdf(pos);
						pos += t * rd;
						hit = t < INTERSECTION_PRECISION;
						//https://docs.microsoft.com/en-us/windows/desktop/direct3dhlsl/dx-graphics-hlsl-any
						if (any(clamp(pos, -BOUND, BOUND) - pos)) {
							break;
						}
						else if (hit)
						{
							break;
						}
					}

					if (hit) {
						float3 normal = calcNormal(pos);

						float filmThickness = fancyCube(iChannel1, normal, THICKNESS_CUBEMAP_SCALE, 0.0).x + 0.1;

						float3 att0 = attenuation(filmThickness, wavelengths0, normal, rd);
						float3 att1 = attenuation(filmThickness, wavelengths1, normal, rd);

						float3 f0 = (1.0 - FRESNEL_RATIO) + FRESNEL_RATIO * fresnel(rd, normal, 1.0 / iors0);
						float3 f1 = (1.0 - FRESNEL_RATIO) + FRESNEL_RATIO * fresnel(rd, normal, 1.0 / iors1);

						float3 rrd = reflect(rd, normal);

						float3 cube0 = REFLECTANCE_GAMMA_SCALE * att0 * sampleCubeMap(wavelengths0, rrd);
						float3 cube1 = REFLECTANCE_GAMMA_SCALE * att1 * sampleCubeMap(wavelengths1, rrd);

						float3 refl0 = REFLECTANCE_SCALE * filmic_gamma_inverse(lerp(float3(0,0,0), cube0, f0));
						float3 refl1 = REFLECTANCE_SCALE * filmic_gamma_inverse(lerp(float3(0,0,0), cube1, f1));

						rds[0] = refract(rd, normal, iors0.x);
						rds[1] = refract(rd, normal, iors0.y);
						rds[2] = refract(rd, normal, iors0.z);
						rds[3] = refract(rd, normal, iors1.x);
						rds[4] = refract(rd, normal, iors1.y);
						rds[5] = refract(rd, normal, iors1.z);

						col += resampleColor(rds, refl0, refl1, wavelengths0, wavelengths1);
					}
			 else {
			  col += resampleColorSimple(rd, wavelengths0, wavelengths1);
			}

			}

			col /= float(AA_SAMPLES);

			return float4(contrast(col), 1.0);
			//return float4(col, 1.0);
            }
            ENDCG
        }
    }
}
