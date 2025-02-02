import { useEffect, useState } from "react";
import Head from "next/head.js";
import { NextRouter, useRouter } from "next/router.js";
import clsx from "clsx";
import {
  useTableOfContents,
  collectHeadings,
  Comments,
  Footer,
  EditThisPage,
  TableOfContents,
  SiteToc,
} from "@portaljs/core";
import type {
  NavItem,
  NavGroup,
  CommentsConfig,
  NavConfig,
  ThemeConfig,
  AuthorConfig,
  TocSection,
} from "@portaljs/core";

interface Props extends React.PropsWithChildren {
  showComments: boolean;
  showEditLink: boolean;
  showSidebar: boolean;
  showToc: boolean;
  nav: NavConfig; // We still accept nav to pull the title and links.
  author: AuthorConfig;
  theme: ThemeConfig;
  urlPath: string;
  commentsConfig: CommentsConfig;
  siteMap: Array<NavItem | NavGroup>;
  editUrl?: string;
}

export const Layout: React.FC<Props> = ({
  children,
  nav,
  author,
  theme,
  showEditLink,
  showToc,
  showSidebar,
  urlPath,
  showComments,
  commentsConfig,
  editUrl,
  siteMap,
}) => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [tableOfContents, setTableOfContents] = useState<TocSection[]>([]);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const currentSection = useTableOfContents(tableOfContents);
  const router: NextRouter = useRouter();

  // Build table of contents on route change if enabled
  useEffect(() => {
    if (!showToc) return;
    const headingNodes: NodeListOf<HTMLHeadingElement> =
      document.querySelectorAll("h1, h2, h3");
    const toc = collectHeadings(headingNodes);
    setTableOfContents(toc ?? []);
  }, [router.asPath, showToc]);

  // Set isScrolled based on window.scrollY
  useEffect(() => {
    function onScroll() {
      setIsScrolled(window.scrollY > 0);
    }
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
    };
  }, []);

  return (
    <>
      <Head>
        <link
          rel="icon"
          href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>💐</text></svg>"
        />
        <meta charSet="utf-8" />
        <meta name="viewport" content="initial-scale=1.0, width=device-width" />
      </Head>

      <div className="min-h-screen bg-background dark:bg-background-dark">
        {/* CUSTOM NAVBAR */}
        <div
          className={clsx(
            "sticky top-0 z-50 w-full",
            isScrolled
              ? "dark:bg-background-dark/95 bg-background/95 backdrop-blur [@supports(backdrop-filter:blur(0))]:dark:bg-background-dark/75"
              : "dark:bg-background-dark bg-background"
          )}
        >
          <div className="h-[4rem] flex items-center justify-between max-w-8xl mx-auto p-4 md:px-8">
            {/* Title */}
            <div className="text-xl font-bold">
              {nav.title}
            </div>

            {/* Right-side Links (if any) */}
            <div className="hidden md:flex space-x-4">
              {nav.links?.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  className="text-sm font-medium hover:text-blue-600"
                >
                  {link.name}
                </a>
              ))}
            </div>

            {/* Hamburger button for off-canvas sidebar toggle on mobile */}
            {showSidebar && (
              <button
                className="md:hidden p-2"
                onClick={() => setMobileSidebarOpen(true)}
                aria-label="Open sidebar"
              >
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              </button>
            )}
          </div>
        </div>

        {/* Mobile Sidebar Overlay */}
        {showSidebar && (
          <div
            className={clsx(
              "md:hidden fixed inset-0 z-40 transition-transform duration-300 ease-in-out",
              mobileSidebarOpen ? "translate-x-0" : "-translate-x-full"
            )}
          >
            {/* Overlay to close the sidebar */}
            <div
              className="absolute inset-0 bg-black opacity-50"
              onClick={() => setMobileSidebarOpen(false)}
            />
            {/* Sidebar content */}
            <div className="relative bg-white dark:bg-background-dark w-64 h-full overflow-y-auto p-4">
              <button
                className="mb-4 p-2"
                onClick={() => setMobileSidebarOpen(false)}
                aria-label="Close sidebar"
              >
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
              <SiteToc currentPath={urlPath} nav={siteMap} />
            </div>
          </div>
        )}

        {/* Desktop Sidebar */}
        {showSidebar && (
          <div className="hidden lg:block fixed z-20 w-[16rem] top-[4rem] right-auto bottom-0 left-[max(0px,calc(50%-44rem))] pt-8 pl-8 overflow-y-auto">
            <SiteToc currentPath={urlPath} nav={siteMap} />
          </div>
        )}

        {/* Wrapper for Main Content and ToC */}
        <div className="max-w-8xl mx-auto px-4 md:px-8">
          <div
            className={clsx(
              "mx-auto lg:px-[16rem] pt-8",
              !showToc && !showSidebar && "lg:px-0"
            )}
          >
            {children}

            {/* Edit This Page Link */}
            {showEditLink && editUrl && <EditThisPage url={editUrl} />}

            {/* Page Comments */}
            {showComments && (
              <div
                className="prose mx-auto pt-6 pb-6 text-center text-gray-700 dark:text-gray-300"
                id="comment"
              >
                <Comments commentsConfig={commentsConfig} slug={urlPath} />
              </div>
            )}
          </div>

          {/* Table of Contents (for extra-large screens) */}
          {showToc && tableOfContents.length > 0 && (
            <div className="hidden xl:block fixed z-20 w-[16rem] top-[4rem] bottom-0 right-[max(0px,calc(50%-44rem))] left-auto pt-8 pr-8 overflow-y-auto">
              <TableOfContents
                tableOfContents={tableOfContents}
                currentSection={currentSection}
              />
            </div>
          )}

          <Footer links={nav.links} author={author} />
        </div>
      </div>
    </>
  );
};
