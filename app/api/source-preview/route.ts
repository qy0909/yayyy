import { NextRequest, NextResponse } from 'next/server';
import http from 'http';
import https from 'https';

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';
const PYTHON_API_TIMEOUT_MS = Number(process.env.PYTHON_API_TIMEOUT_MS || 600000);

type PythonApiResponse = {
  status: number;
  data: unknown;
};

function callPythonApi(path: string, method: 'GET'): Promise<PythonApiResponse> {
  const baseUrl = new URL(PYTHON_API_URL);
  const isHttps = baseUrl.protocol === 'https:';
  const client = isHttps ? https : http;
  const requestPath = `${baseUrl.pathname.replace(/\/$/, '')}${path}`;

  return new Promise((resolve, reject) => {
    const req = client.request(
      {
        protocol: baseUrl.protocol,
        hostname: baseUrl.hostname,
        port: baseUrl.port || (isHttps ? 443 : 80),
        path: requestPath,
        method,
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
            resolve({ status, data: raw ? JSON.parse(raw) : null });
          } catch {
            resolve({ status, data: { raw } });
          }
        });
      }
    );

    req.setTimeout(PYTHON_API_TIMEOUT_MS, () => {
      req.destroy(new Error(`Python API timeout after ${PYTHON_API_TIMEOUT_MS}ms`));
    });

    req.on('error', reject);
    req.end();
  });
}

export async function GET(request: NextRequest) {
  try {
    const sourceUrl = request.nextUrl.searchParams.get('source_url');
    const highlightChunkIndex = request.nextUrl.searchParams.get('highlight_chunk_index');
    const highlightTitle = request.nextUrl.searchParams.get('highlight_title');

    if (!sourceUrl) {
      return NextResponse.json({ error: 'source_url is required' }, { status: 400 });
    }

    const params = new URLSearchParams({ source_url: sourceUrl });
    if (highlightChunkIndex !== null && highlightChunkIndex !== '') {
      params.set('highlight_chunk_index', highlightChunkIndex);
    }
    if (highlightTitle !== null && highlightTitle !== '') {
      params.set('highlight_title', highlightTitle);
    }

    const response = await callPythonApi(`/api/source-preview?${params.toString()}`, 'GET');

    if (response.status < 200 || response.status >= 300) {
      const message = (response.data as { detail?: string })?.detail || `Python API returned ${response.status}`;
      throw new Error(message);
    }

    return NextResponse.json(response.data);
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Failed to load source preview',
      },
      { status: 500 }
    );
  }
}
