import { NextRequest, NextResponse } from 'next/server';
import http from 'http';
import https from 'https';

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';
const PYTHON_API_TIMEOUT_MS = Number(process.env.PYTHON_API_TIMEOUT_MS || 600000);

type PythonApiResponse = {
  status: number;
  data: unknown;
};

function callPythonApi(path: string, method: 'PATCH', body?: unknown, headers?: Record<string, string>): Promise<PythonApiResponse> {
  const baseUrl = new URL(PYTHON_API_URL);
  const isHttps = baseUrl.protocol === 'https:';
  const client = isHttps ? https : http;

  const payload = body ? JSON.stringify(body) : undefined;
  const requestPath = `${baseUrl.pathname.replace(/\/$/, '')}${path}`;

  return new Promise((resolve, reject) => {
    const req = client.request(
      {
        protocol: baseUrl.protocol,
        hostname: baseUrl.hostname,
        port: baseUrl.port || (isHttps ? 443 : 80),
        path: requestPath,
        method,
        headers: {
          'Content-Type': 'application/json',
          ...(payload ? { 'Content-Length': Buffer.byteLength(payload).toString() } : {}),
          ...(headers || {}),
        },
      },
      (res) => {
        const chunks: Buffer[] = [];

        res.on('data', (chunk) => {
          chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
        });

        res.on('end', () => {
          const raw = Buffer.concat(chunks).toString('utf-8');
          const status = res.statusCode || 500;

          try {
            const parsed = raw ? JSON.parse(raw) : null;
            resolve({ status, data: parsed });
          } catch {
            resolve({ status, data: { raw } });
          }
        });
      }
    );

    req.setTimeout(PYTHON_API_TIMEOUT_MS, () => {
      req.destroy(new Error(`Python API timeout after ${PYTHON_API_TIMEOUT_MS}ms`));
    });

    req.on('error', (error) => {
      reject(error);
    });

    if (payload) {
      req.write(payload);
    }

    req.end();
  });
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ submissionId: string }> }
) {
  try {
    const body = await request.json();
    const { submissionId } = await params;
    const adminToken = request.headers.get('x-admin-token') || '';

    const response = await callPythonApi(
      `/api/slang-terms/submissions/${encodeURIComponent(submissionId)}/review`,
      'PATCH',
      body,
      adminToken ? { 'x-admin-token': adminToken } : undefined
    );

    return NextResponse.json(response.data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to review submission',
      },
      { status: 500 }
    );
  }
}
