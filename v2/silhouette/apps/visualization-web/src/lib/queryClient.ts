import { QueryClient } from "@tanstack/react-query";

/** 필터가 queryKey에 포함되므로, 같은 필터로 탭을 왔다 갔다 할 때 불필요한 재요청을 줄인다. */
const DEFAULT_STALE_MS = 5 * 60 * 1000;
/** 탭 전환 시 캐시가 너무 빨리 버려지지 않도록 기본 gcTime을 늘린다. */
const DEFAULT_GC_MS = 30 * 60 * 1000;

export function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: DEFAULT_STALE_MS,
        gcTime: DEFAULT_GC_MS,
        refetchOnWindowFocus: false,
        retry: 1,
      },
    },
  });
}
