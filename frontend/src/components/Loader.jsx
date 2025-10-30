import React from "react";

export default function Loader() {
  return (
    <div className="flex justify-start my-3">
      <div className="flex gap-1">
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-150"></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-300"></div>
      </div>
    </div>
  );
}
