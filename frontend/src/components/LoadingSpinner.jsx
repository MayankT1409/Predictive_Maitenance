export default function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center py-10">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-gray-300 border-t-blue-600" />
    </div>
  );
}
