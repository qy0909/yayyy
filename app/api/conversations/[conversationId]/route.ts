import { NextRequest, NextResponse } from 'next/server';
import http from 'http';
import https from 'https';

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';
const PYTHON_API_TIMEOUT_MS = Number(process.env.PYTHON_API_TIMEOUT_MS || 600000);

type PythonApiResponse = {
  status: number;
  data: unknown;
};

function callPythonApi(path: string, method: 'GET' | 'DELETE', sessionId?: string): Promise<PythonApiResponse> {
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
        headers: {
          'Content-Type': 'application/json',
          'x-session-id': sessionId || 'anonymous-session',
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

export async function GET(request: NextRequest, props: { params: Promise<{ conversationId: string }> }) {
  const params = await props.params;
  if (!params.conversationId || params.conversationId === 'undefined' || params.conversationId === 'null') {
    return NextResponse.json({ error: 'Invalid conversation ID' }, { status: 400 });
  }
  
  const sessionId = request.headers.get('x-session-id') || 'anonymous-session';
  const response = await callPythonApi(`/api/conversations/${params.conversationId}`, 'GET', sessionId);
  return NextResponse.json(response.data, { status: response.status });
}

export async function DELETE(request: NextRequest, props: { params: Promise<{ conversationId: string }> }) {
  const params = await props.params;
  if (!params.conversationId || params.conversationId === 'undefined' || params.conversationId === 'null') {
    return NextResponse.json({ error: 'Invalid conversation ID' }, { status: 400 });
  }

  const sessionId = request.headers.get('x-session-id') || 'anonymous-session';
  const response = await callPythonApi(`/api/conversations/${params.conversationId}`, 'DELETE', sessionId);
  return NextResponse.json(response.data, { status: response.status });
}