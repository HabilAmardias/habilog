export function formatDate(date) {
	const dateToFormat = new Date(date.replaceAll('-', '/'))
	const dateFormatter = new Intl.DateTimeFormat('en', { dateStyle:'medium' })
	return dateFormatter.format(dateToFormat)
}

export function paginate(data) {
	const pages = [];
	for (let i = 0; i < data.length; i += 8) {
	  pages.push(data.slice(i, i + 8));
	}
	return pages;
  }