const SUPPORTED_EXT_RE = /\.(png|jpe?g|webp|gif|avif|cbz|zip)$/i;

export function isSupportedComicFile(name: string): boolean {
  const base = name.split(/[\\/]/).pop() || name;
  if (!base || base.startsWith(".")) return false;
  return SUPPORTED_EXT_RE.test(base);
}

export function naturalSortFiles(files: File[]): File[] {
  return [...files].sort((a, b) => {
    const ka = (a as File & { webkitRelativePath?: string }).webkitRelativePath || a.name;
    const kb = (b as File & { webkitRelativePath?: string }).webkitRelativePath || b.name;
    return ka.localeCompare(kb, undefined, { numeric: true, sensitivity: "base" });
  });
}

export function filterAndSortComicFiles(files: File[]): File[] {
  return naturalSortFiles(files.filter((f) => isSupportedComicFile(f.name)));
}

export function inferFolderOrFileTitle(files: File[], explicitFolderName = ""): string {
  if (explicitFolderName.trim()) return explicitFolderName.trim();
  if (files.length === 0) return "";
  for (const f of files) {
    const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath || "";
    if (rel.includes("/")) {
      const top = rel.split("/")[0]?.trim();
      if (top) return top;
    }
  }
  const stem = files[0].name.replace(/\.[^/.]+$/, "");
  return stem.replace(/[_-]+/g, " ").trim();
}

type EntryLike = {
  isFile: boolean;
  isDirectory: boolean;
  name: string;
  file?: (cb: (f: File) => void, err?: (e: unknown) => void) => void;
  createReader?: () => {
    readEntries: (cb: (entries: EntryLike[]) => void, err?: (e: unknown) => void) => void;
  };
};

async function readAllDirEntries(entry: EntryLike): Promise<EntryLike[]> {
  if (!entry.createReader) return [];
  const reader = entry.createReader();
  const all: EntryLike[] = [];
  for (;;) {
    const batch = await new Promise<EntryLike[]>((resolve) => {
      reader.readEntries(
        (entries) => resolve(entries || []),
        () => resolve([])
      );
    });
    if (batch.length === 0) break;
    all.push(...batch);
  }
  return all;
}

async function traverseEntry(entry: EntryLike, prefix: string): Promise<File[]> {
  if (entry.isFile && entry.file) {
    return new Promise<File[]>((resolve) => {
      entry.file!(
        (f) => {
          if (!isSupportedComicFile(f.name)) {
            resolve([]);
            return;
          }
          // Attach relative path metadata for natural sort & title inference
          const relPath = prefix ? `${prefix}/${f.name}` : f.name;
          try {
            Object.defineProperty(f, "webkitRelativePath", {
              value: relPath,
              configurable: true,
            });
          } catch {
            // read-only on some engines; safe to ignore
          }
          resolve([f]);
        },
        () => resolve([])
      );
    });
  }
  if (entry.isDirectory) {
    const children = await readAllDirEntries(entry);
    const nextPrefix = prefix ? `${prefix}/${entry.name}` : entry.name;
    const nested = await Promise.all(children.map((ch) => traverseEntry(ch, nextPrefix)));
    return nested.flat();
  }
  return [];
}

/**
 * Extracts all supported comic images / archives from a DragEvent's DataTransfer,
 * recursively descending into any dropped folders via webkitGetAsEntry().
 */
export async function extractFilesFromDataTransfer(
  dt: DataTransfer | null
): Promise<{ files: File[]; folderName: string }> {
  if (!dt) return { files: [], folderName: "" };
  const items = dt.items ? Array.from(dt.items) : [];
  const entries: EntryLike[] = [];
  let folderName = "";

  for (const item of items) {
    if (item.kind !== "file") continue;
    const fn = (item as DataTransferItem & { webkitGetAsEntry?: () => EntryLike | null })
      .webkitGetAsEntry;
    if (typeof fn === "function") {
      const entry = fn.call(item);
      if (entry) {
        if (entry.isDirectory && !folderName) {
          folderName = entry.name;
        }
        entries.push(entry);
      }
    }
  }

  if (entries.length > 0) {
    const collected = (await Promise.all(entries.map((e) => traverseEntry(e, "")))).flat();
    return { files: filterAndSortComicFiles(collected), folderName };
  }

  const rawFiles = dt.files ? Array.from(dt.files) : [];
  return { files: filterAndSortComicFiles(rawFiles), folderName: "" };
}
